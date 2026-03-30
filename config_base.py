@dataclass
class ModelConfig:
    num_classes: int = 3

    # text # TODO
    bert_path: str = "./finetuned_bert_mlm"
    text_dropout: float = 0.3
    freeze_text: bool = False

    # audio
    wav2vec_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    audio_dropout: float = 0.3
    freeze_audio: bool = False

    # video # TODO
    video_checkpoint: str = "./checkpoints/video_model.pt"
    freeze_video: bool = False
    video_face_hidden: int = 128
    video_pose_hidden: int = 64

    # fusion
    fusion_dropout: float = 0.3
