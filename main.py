from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config import get_config
from datasets.multimodal_classification import MultimodalClassificationDataset
from utils.collate import multimodal_collate_fn
from models.text import BertTextClassifier
from models.audio import AudioPlaceholderClassifier, Wav2Vec2Classifier
from models.fusion import LateFusionModel, HiddenFusionModel
from models.video_adapter import VideoClassifierAdapter
from train import train_model

from ../Video/cspan_congressionalrhetoric_video/training/models import DualStreamEncoder


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

    if cfg.model.use_audio_placeholder:
        audio_model = AudioPlaceholderClassifier(
            hidden_dim=cfg.model.audio_hidden_dim,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.audio_dropout,
        )
    else:
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

    if cfg.model.fusion_type == "late":
        model = LateFusionModel(
            video_model=video_model,
            text_model=text_model,
            audio_model=audio_model,
            num_classes=cfg.model.num_classes,
            fusion_hidden=32,
            dropout=cfg.model.fusion_dropout,
        )
    elif cfg.model.fusion_type == "hidden":
        model = HiddenFusionModel(
            video_model=video_model,
            text_model=text_model,
            audio_model=audio_model,
            num_classes=cfg.model.num_classes,
            fusion_hidden=cfg.model.fusion_hidden_dim,
            dropout=cfg.model.fusion_dropout,
        )
    else:
        raise ValueError(f"Unsupported fusion_type: {cfg.model.fusion_type}")

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