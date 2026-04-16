# configs/unimodal_audio.py
# Train with audio modality only.
# Run with: python main.py --config configs.unimodal_audio --modality audio

from config_base import *


def get_config():
    return FullConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(
            freeze_text=True,
            freeze_audio=False,
            freeze_video=True,
        ),
        train=TrainConfig(
            epochs=15,
            batch_size=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            patience=5,
            save_dir="./outputs/unimodal_audio",
        ),
    )
