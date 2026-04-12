import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from config_base import FullConfig
from datasets.multimodal_classification import MultimodalClassificationDataset
from models.audio import ChunkedWav2VecAudioEncoder
from models.fuse import CrossModalAttentionFusion, MultimodalFusionModel
from models.text import BertTextClassifier
from models.video import VideoClassifierAdapter
from utils.collate import multimodal_collate_fn
from utils.metrics import confusion_matrix_torch, macro_f1_from_confusion_matrix


def _add_video_repo_to_path() -> Path:
    env_path = os.environ.get("VIDEO_REPO_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        Path(__file__).resolve().parent.parent
        / "Video"
        / "cspan_congressionalrhetoric_video",
    ]
    for c in candidates:
        if c and c.exists():
            sys.path.insert(0, str(c))
            return c
    raise RuntimeError(
        "Video repo not found. Set the VIDEO_REPO_PATH environment variable."
    )


_add_video_repo_to_path()
from training.models import DualStreamEncoder  # noqa: E402


def build_model_from_config(cfg: FullConfig) -> MultimodalFusionModel:
    text_model = BertTextClassifier(
        pretrained_path=cfg.model.bert_path,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.text_dropout,
        freeze=False,
    )
    audio_model = ChunkedWav2VecAudioEncoder(
        model_name=cfg.model.wav2vec_name,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.audio_dropout,
        freeze_backbone=False,
    )
    raw_video_model = DualStreamEncoder(
        face_hidden=cfg.model.video_face_hidden,
        pose_hidden=cfg.model.video_pose_hidden,
        num_classes=cfg.model.num_classes,
        freeze_backbone=False,
    )
    video_model = VideoClassifierAdapter(
        raw_video_model,
        checkpoint_path=cfg.model.video_checkpoint,
        freeze=False,
    )
    fusion = CrossModalAttentionFusion(
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.fusion_dropout,
    )
    return MultimodalFusionModel(
        video_encoder=video_model,
        video_dim=video_model.output_dim,
        text_encoder=text_model,
        text_dim=text_model.output_dim,
        audio_encoder=audio_model,
        audio_dim=audio_model.output_dim,
        fusion=fusion,
    )


def load_model(checkpoint_path: str, device: torch.device) -> MultimodalFusionModel:
    """
    Load a model from a training checkpoint.

    The checkpoint must contain either:
        - 'config': the serialised FullConfig dataclass  (preferred)
        - 'model_state_dict': raw state dict

    If the checkpoint does not embed a config, a default FullConfig() is used –
    make sure the architecture matches.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config
    cfg_dict = ckpt.get("config", None)
    if cfg_dict is not None and isinstance(cfg_dict, FullConfig):
        cfg = cfg_dict
    else:
        print("[WARNING] No config found in checkpoint – using default FullConfig().")
        from config_base import DatasetConfig, ModelConfig, TrainConfig

        cfg = FullConfig(DatasetConfig(), ModelConfig(), TrainConfig())

    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    val_f1 = metrics.get("val_f1_macro", "?")
    print(f"Loaded checkpoint  epoch={epoch}  val_f1={val_f1}")

    return model, cfg


def build_dataloader(
    cfg: FullConfig,
    split: str = "val",
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = BertTokenizer.from_pretrained(cfg.model.bert_path)
    ds = MultimodalClassificationDataset(
        text_dir=cfg.dataset.text_dir,
        video_dir=cfg.dataset.video_dir,
        audio_dir=cfg.dataset.audio_dir,
        tokenizer=tokenizer,
        split=split,
        max_text_length=cfg.dataset.max_text_length,
        audio_sample_rate=cfg.dataset.audio_sample_rate,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )


LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def _move_batch(batch: dict, device: torch.device) -> dict:
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


def predict_dataloader(
    model: MultimodalFusionModel,
    dataloader: DataLoader,
    device: torch.device,
    modality: Optional[str] = None,
) -> dict:
    """
    Run inference over a DataLoader.

    Returns a dict with:
        preds        [N] int tensor of predicted class indices
        labels       [N] int tensor of ground-truth labels
        probs        [N, C] float tensor of softmax probabilities
        metrics      dict with loss, acc, f1_macro, confusion_matrix
    """
    all_preds, all_labels, all_probs = [], [], []
    num_classes = model.num_classes

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference [{modality or 'multimodal'}]"):
            batch = _move_batch(batch, device)

            if modality is None:
                out = model(
                    faces=batch["video"]["faces"],
                    pose=batch["video"]["pose"],
                    lengths=batch["video"]["lengths"],
                    input_ids=batch["text"]["input_ids"],
                    attention_mask=batch["text"]["attention_mask"],
                    waveform=batch["audio"]["waveform"],
                    audio_attention_mask=batch["audio"]["attention_mask"],
                )
            else:
                out = model.forward_unimodal(
                    modality=modality,
                    faces=batch["video"]["faces"],
                    pose=batch["video"]["pose"],
                    lengths=batch["video"]["lengths"],
                    input_ids=batch["text"]["input_ids"],
                    attention_mask=batch["text"]["attention_mask"],
                    waveform=batch["audio"]["waveform"],
                    audio_attention_mask=batch["audio"]["attention_mask"],
                )

            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(batch["label"].cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    cm = confusion_matrix_torch(all_preds, all_labels, num_classes)
    acc = (all_preds == all_labels).float().mean().item()
    f1 = macro_f1_from_confusion_matrix(cm)

    return {
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
        "metrics": {
            "acc": acc,
            "f1_macro": f1,
            "confusion_matrix": cm.tolist(),
        },
    }


def print_results(results: dict, modality: Optional[str] = None) -> None:
    m = results["metrics"]
    tag = modality or "multimodal"
    print(f"\n{'─' * 50}")
    print(f"  Results  [{tag}]")
    print(f"{'─' * 50}")
    print(f"  Accuracy  : {m['acc']:.4f}")
    print(f"  Macro F1  : {m['f1_macro']:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    cm = m["confusion_matrix"]
    header = "          " + "  ".join(f"{LABEL_NAMES[i]:>8}" for i in range(len(cm)))
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {LABEL_NAMES[i]:>8}  {row_str}")
    print(f"{'─' * 50}\n")


def run_ablation(
    checkpoint_path: str,
    split: str = "val",
    batch_size: int = 4,
    device_str: str = "cuda",
    output_json: Optional[str] = None,
) -> dict:
    """
    Run multimodal + all three unimodal variants and compare them.

    Returns a summary dict and optionally writes it to a JSON file.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint_path, device)
    loader = build_dataloader(cfg, split=split, batch_size=batch_size)

    summary = {}
    for mod in [None, "text", "audio", "video"]:
        tag = mod or "multimodal"
        print(f"\n>>> Running {tag} …")
        res = predict_dataloader(model, loader, device, modality=mod)
        print_results(res, modality=mod)
        summary[tag] = res["metrics"]

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Ablation summary written to {output_json}")

    return summary


class Predictor:
    """
    High-level predictor that wraps a loaded checkpoint.

    Example::

        p = Predictor("outputs/best.pt")
        results = p.predict_split("test")
        p.print_results(results)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device_str: str = "cuda",
        batch_size: int = 4,
    ) -> None:
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = load_model(checkpoint_path, self.device)
        self.batch_size = batch_size

    def predict_split(
        self,
        split: str = "val",
        modality: Optional[str] = None,
    ) -> dict:
        loader = build_dataloader(
            self.cfg,
            split=split,
            batch_size=self.batch_size,
        )
        return predict_dataloader(self.model, loader, self.device, modality=modality)

    def predict_batch(
        self,
        batch: dict,
        modality: Optional[str] = None,
    ) -> dict:
        """
        Run inference on a pre-collated batch dict.
        """
        batch = _move_batch(batch, self.device)
        with torch.no_grad():
            if modality is None:
                out = self.model(
                    faces=batch["video"]["faces"],
                    pose=batch["video"]["pose"],
                    lengths=batch["video"]["lengths"],
                    input_ids=batch["text"]["input_ids"],
                    attention_mask=batch["text"]["attention_mask"],
                    waveform=batch["audio"]["waveform"],
                    audio_attention_mask=batch["audio"]["attention_mask"],
                )
            else:
                out = self.model.forward_unimodal(
                    modality=modality,
                    faces=batch["video"]["faces"],
                    pose=batch["video"]["pose"],
                    lengths=batch["video"]["lengths"],
                    input_ids=batch["text"]["input_ids"],
                    attention_mask=batch["text"]["attention_mask"],
                    waveform=batch["audio"]["waveform"],
                    audio_attention_mask=batch["audio"]["attention_mask"],
                )
        probs = torch.softmax(out["logits"], dim=-1)
        preds = out["logits"].argmax(dim=-1)
        return {
            "preds": preds.cpu(),
            "probs": probs.cpu(),
            "label_names": [LABEL_NAMES[p.item()] for p in preds.cpu()],
        }

    @staticmethod
    def print_results(results: dict, modality: Optional[str] = None) -> None:
        print_results(results, modality=modality)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference / ablation from a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best.pt or last.pt checkpoint file.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--modality",
        default=None,
        choices=["text", "audio", "video"],
        help="Run unimodal evaluation for this modality only.  "
        "Omit for full multimodal evaluation.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run all modality combinations and print a comparison table.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write ablation results as JSON.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    args = parser.parse_args()

    if args.ablation:
        run_ablation(
            checkpoint_path=args.checkpoint,
            split=args.split,
            batch_size=args.batch_size,
            device_str=args.device,
            output_json=args.output_json,
        )
        return

    # Single-mode evaluation
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    loader = build_dataloader(cfg, split=args.split, batch_size=args.batch_size)
    results = predict_dataloader(model, loader, device, modality=args.modality)
    print_results(results, modality=args.modality)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results["metrics"], f, indent=2)
        print(f"Metrics written to {args.output_json}")


if __name__ == "__main__":
    main()
