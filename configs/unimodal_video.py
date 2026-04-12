# configs/unimodal_video.py
# Train with video modality only.
# Run with: python main.py --config configs.unimodal_video --modality video

from config_base import *


def get_config():
    return FullConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(
            freeze_text=True,
            freeze_audio=True,
            freeze_video=False,
        ),
        train=TrainConfig(
            epochs=15,
            batch_size=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            patience=5,
            save_dir="./outputs/unimodal_video",
        ),
    )
