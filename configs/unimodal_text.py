# configs/unimodal_text.py
# Train with text modality only.
# Run with: python main.py --config configs.unimodal_text --modality text

from config_base import *


def get_config():
    return FullConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(
            freeze_text=False,
            freeze_audio=True,
            freeze_video=True,
        ),
        train=TrainConfig(
            epochs=15,
            batch_size=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            patience=5,
            save_dir="./outputs/unimodal_text",
        ),
    )
