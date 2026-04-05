import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    text_dir: str = "../dataset"
    video_dir: str = (
        "../Video/cspan_congressionalrhetoric_video/data/processed/frame_skip_30"
    )
    audio_dir: str = "../dataset/raw_audio"
    max_text_length: int = 256
    audio_sample_rate: int = 16000
    skip_validation: bool = False


@dataclass
class ModelConfig:
    num_classes: int = 3

    # text
    bert_path: str = "pretrained/bert_mlm"
    text_dropout: float = 0.3
    freeze_text: bool = True

    # audio
    wav2vec_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    audio_dropout: float = 0.3
    freeze_audio: bool = True

    # video
    video_checkpoint: str = "pretrained/video/dual_stream_best.pt"
    freeze_video: bool = True
    video_face_hidden: int = 128
    video_pose_hidden: int = 64

    # fusion
    fusion_dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    save_dir: str = "./outputs"
    run_name: str = ""
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    grad_clip: float = 1.0


@dataclass
class FullConfig:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig

    def save_json(self, path: str | Path) -> None:
        payload = asdict(self)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
