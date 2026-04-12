import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
    grad_clip: float = 1.0,
    scaler=None,
    modality: Optional[str] = None,
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
                if modality is None:
                    outputs = model(
                        faces=batch["video"]["faces"],
                        pose=batch["video"]["pose"],
                        lengths=batch["video"]["lengths"],
                        input_ids=batch["text"]["input_ids"],
                        attention_mask=batch["text"]["attention_mask"],
                        waveform=batch["audio"]["waveform"],
                        audio_attention_mask=batch["audio"]["attention_mask"],
                    )
                else:
                    outputs = model.forward_unimodal(
                        modality=modality,
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
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    save_dir: Path,
    name: str = "best.pt",
    cfg=None,
):
    """Save a full training checkpoint AND a lean weights-only file."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": cfg,
        },
        ckpt_path,
    )

    weights_name = name.replace(".pt", "_weights.pt")
    torch.save(model.state_dict(), save_dir / weights_name)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


def _build_optimizer(model, cfg, modality: Optional[str] = None):
    base_lr = cfg.train.learning_rate
    encoder_lr = base_lr * 0.1

    if modality == "text":
        param_groups = [
            {"params": list(model.text_encoder.parameters()), "lr": encoder_lr},
            {"params": list(model.text_proj.parameters()), "lr": base_lr},
            {"params": list(model.fusion.parameters()), "lr": base_lr},
        ]
    elif modality == "audio":
        param_groups = [
            {"params": list(model.audio_encoder.parameters()), "lr": encoder_lr},
            {"params": list(model.audio_proj.parameters()), "lr": base_lr},
            {"params": list(model.fusion.parameters()), "lr": base_lr},
        ]
    elif modality == "video":
        param_groups = [
            {"params": list(model.video_encoder.parameters()), "lr": encoder_lr},
            {"params": list(model.video_proj.parameters()), "lr": base_lr},
            {"params": list(model.fusion.parameters()), "lr": base_lr},
        ]
    else:
        param_groups = [
            {
                "params": [
                    p for p in model.text_encoder.parameters() if p.requires_grad
                ],
                "lr": encoder_lr,
            },
            {
                "params": [
                    p for p in model.audio_encoder.parameters() if p.requires_grad
                ],
                "lr": encoder_lr,
            },
            {
                "params": (
                    list(model.video_proj.parameters())
                    + list(model.text_proj.parameters())
                    + list(model.audio_proj.parameters())
                ),
                "lr": base_lr,
            },
            {
                "params": list(model.fusion.parameters()),
                "lr": base_lr,
            },
        ]

    return AdamW(param_groups, weight_decay=cfg.train.weight_decay)


def train_model(
    model,
    train_loader,
    val_loader,
    cfg,
    resume_from: Optional[str] = None,
    modality: Optional[str] = None,
):
    """
    Train (or fine-tune) the model.

    Args:
        modality: 'text' | 'audio' | 'video' | None.
                  When set, only that modality's encoder + projection + fusion
                  head are updated; the remaining encoder branches are inactive.
    """
    set_seed(cfg.train.seed)

    if modality is not None:
        assert modality in {"text", "audio", "video"}, (
            f"modality must be 'text', 'audio', 'video', or None – got '{modality}'"
        )

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = _build_optimizer(model, cfg, modality)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=1e-7,
    )

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

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    scaler = GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    history = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = getattr(cfg.train, "run_name", "").strip()
    if modality:
        run_name = f"{run_name}_{modality}" if run_name else modality
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

    patience = getattr(cfg.train, "patience", 5)
    epochs_no_improve = 0

    mode_tag = f"[{modality} only]" if modality else "[multimodal]"
    print(f"\nTraining mode: {mode_tag}")

    for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.train.epochs} {mode_tag} =====")

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.train.grad_clip,
            scaler=scaler,
            modality=modality,
        )

        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            modality=modality,
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "modality": modality or "multimodal",
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_confusion_matrix": val_metrics["confusion_matrix"].tolist(),
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        improved = val_metrics["f1_macro"] > best_val_f1
        if improved:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                record,
                save_dir,
                name="best.pt",
                cfg=cfg,
            )
            print(
                f"  ✓ New best val F1: {best_val_f1:.4f} → saved best.pt + best_weights.pt"
            )
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            record,
            save_dir,
            name="last.pt",
            cfg=cfg,
        )

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    print(f"\nBest val F1 = {best_val_f1:.4f}  |  outputs saved to {save_dir}")
    return model, history
