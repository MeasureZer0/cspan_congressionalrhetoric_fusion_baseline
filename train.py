from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
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


def run_epoch(model, dataloader, criterion, device, optimizer=None, grad_clip=1.0):
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
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
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


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, name="best.pt"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
        },
        ckpt_path,
    )


def train_model(model, train_loader, val_loader, cfg):
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(save_dir / "config.json")

    history = []
    best_val_f1 = -1.0

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.train.epochs} =====")

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.train.grad_clip,
        )

        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        record = {
            "epoch": epoch,
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

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            save_checkpoint(model, optimizer, epoch, record, save_dir, name="best.pt")

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return model, history