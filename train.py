import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.metrics import confusion_matrix_torch, macro_f1_from_confusion_matrix


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {
        "text": {
            "input_ids": batch["text"]["input_ids"].to(device),
            "attention_mask": batch["text"]["attention_mask"].to(device),
        },
        "audio": {
            "waveform": batch["audio"]["waveform"].to(device),
            "attention_mask": batch["audio"]["attention_mask"].to(device),
            "lengths": batch["audio"]["lengths"].to(device),
        },
        "video": {
            "faces": batch["video"]["faces"].to(device),
            "pose": batch["video"]["pose"].to(device),
            "lengths": batch["video"]["lengths"].to(device),
        },
        "label": batch["label"].to(device),
    }


def run_epoch(
    model, dataloader, criterion, device, optimizer=None, grad_clip=1.0, scaler=None
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Train" if is_train else "Eval")

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        with torch.set_grad_enabled(is_train):
            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(
                    faces=batch["video"]["faces"],
                    pose=batch["video"]["pose"],
                    lengths=batch["video"]["lengths"],
                    input_ids=batch["text"]["input_ids"],
                    attention_mask=batch["text"]["attention_mask"],
                    waveform=batch["audio"]["waveform"],
                    audio_attention_mask=batch["audio"]["attention_mask"],
                )
                logits = outputs["logits"]
                loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        preds = logits.argmax(dim=-1)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())

        pbar.set_postfix({"loss": f"{total_loss / max(total_count, 1):.4f}"})

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    cm = confusion_matrix_torch(all_preds, all_targets, num_classes=model.num_classes)
    acc = (all_preds == all_targets).float().mean().item()
    f1 = macro_f1_from_confusion_matrix(cm)

    return {
        "loss": total_loss / total_count,
        "acc": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
    }


def save_checkpoint(
    model, optimizer, scheduler, epoch, metrics, save_dir, name="best.pt"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "metrics": metrics,
        },
        ckpt_path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


def train_model(model, train_loader, val_loader, cfg, resume_from: str | None = None):
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(
        [
            {
                "params": [
                    p for p in model.text_encoder.parameters() if p.requires_grad
                ],
                "lr": 2e-6,
            },
            {
                "params": [
                    p for p in model.audio_encoder.parameters() if p.requires_grad
                ],
                "lr": 2e-6,
            },
            {
                "params": list(model.video_proj.parameters())
                + list(model.text_proj.parameters())
                + list(model.audio_proj.parameters()),
                "lr": 2e-5,
            },
            {
                "params": list(model.fusion.parameters()),
                "lr": 2e-5,
            },
        ],
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=1e-7,
    )

    # Class weights for imbalanced data
    all_labels = []
    for batch in train_loader:
        all_labels.append(batch["label"])
    all_labels = torch.cat(all_labels, dim=0)
    class_counts = torch.bincount(all_labels, minlength=cfg.model.num_classes).float()
    class_weights = 1.0 / class_counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * cfg.model.num_classes
    class_weights = class_weights.to(device)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    # Label smoothing reduces overconfidence on majority class
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    scaler = GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    history = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = getattr(cfg.train, "run_name", "").strip()
    run_label = f"{run_name}_{run_id}" if run_name else run_id
    save_dir = Path(cfg.train.save_dir) / run_label
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(save_dir / "config.json")

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch = load_checkpoint(resume_from, model, optimizer, scheduler)
        history_path = Path(resume_from).parent / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        print(f"Resuming from epoch {start_epoch + 1}")

    best_val_f1 = max((r["val_f1_macro"] for r in history), default=-1.0)

    # Early stopping state
    patience = getattr(cfg.train, "patience", 5)
    epochs_no_improve = 0

    for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.train.epochs} =====")
        print(f"  LR (fusion): {scheduler.get_last_lr()}")

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.train.grad_clip,
            scaler=scaler,
        )

        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_confusion_matrix": val_metrics["confusion_matrix"].tolist(),
            "lr": scheduler.get_last_lr(),
        }
        history.append(record)

        print(json.dumps(record, indent=2))

        improved = val_metrics["f1_macro"] > best_val_f1
        if improved:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, record, save_dir, name="best.pt"
            )
            print(f"  ✓ New best val F1: {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")

        save_checkpoint(
            model, optimizer, scheduler, epoch, record, save_dir, name="last.pt"
        )

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)"
            )
            break

    return model, history
