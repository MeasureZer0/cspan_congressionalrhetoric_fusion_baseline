import argparse
import importlib
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from datasets.multimodal_classification import MultimodalClassificationDataset
from models.audio import Wav2Vec2Classifier
from models.fuse import CrossModalAttentionFusion, MultimodalFusionModel
from models.text import BertTextClassifier
from models.video import VideoClassifierAdapter
from train import train_model
from utils.collate import multimodal_collate_fn


def _add_video_repo_to_path() -> Path:
    env_path = os.environ.get("VIDEO_REPO_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        Path(__file__).resolve().parent.parent
        / "f2025_s2026_wl_cspan_congressionalrhetoric_video",
        Path(
            "/home/x-jlewandowski/corporate/ccspan-congressional/Video/"
            "cspan_congressionalrhetoric_video"
        ),
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            sys.path.insert(0, str(candidate))
            return candidate

    raise RuntimeError(
        "Video repo not found. Set VIDEO_REPO_PATH or place it next to fusion_baseline."
    )


_add_video_repo_to_path()
from training.models import DualStreamEncoder


def build_dataloaders(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.model.bert_path)

    train_ds = MultimodalClassificationDataset(
        text_dir=cfg.dataset.text_dir,
        video_dir=cfg.dataset.video_dir,
        audio_dir=cfg.dataset.audio_dir,
        tokenizer=tokenizer,
        split="train",
        max_text_length=cfg.dataset.max_text_length,
        audio_sample_rate=cfg.dataset.audio_sample_rate,
        skip_validation=cfg.dataset.skip_validation,
    )

    val_ds = MultimodalClassificationDataset(
        text_dir=cfg.dataset.text_dir,
        video_dir=cfg.dataset.video_dir,
        audio_dir=cfg.dataset.audio_dir,
        tokenizer=tokenizer,
        split="val",
        max_text_length=cfg.dataset.max_text_length,
        audio_sample_rate=cfg.dataset.audio_sample_rate,
        skip_validation=cfg.dataset.skip_validation,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model(cfg):
    text_model = BertTextClassifier(
        pretrained_path=cfg.model.bert_path,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.text_dropout,
        freeze=cfg.model.freeze_text,
    )

    audio_model = Wav2Vec2Classifier(
        model_name=cfg.model.wav2vec_name,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.audio_dropout,
        freeze=cfg.model.freeze_audio,
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
        freeze=cfg.model.freeze_video,
    )

    fusion_model = CrossModalAttentionFusion(
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.fusion_dropout,
    )

    model = MultimodalFusionModel(
        video_encoder=video_model,
        video_dim=video_model.output_dim,
        text_encoder=text_model,
        text_dim=text_model.output_dim,
        audio_encoder=audio_model,
        audio_dim=audio_model.output_dim,
        fusion=fusion_model,
    )

    return model


def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.get_config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs.baseline_hidden",
        help="config module path",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    train_model(model, train_loader, val_loader, cfg)


if __name__ == "__main__":
    main()
