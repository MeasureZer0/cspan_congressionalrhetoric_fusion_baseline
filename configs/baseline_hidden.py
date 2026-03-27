from config_base import *

def get_config():
    return FullConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(
            fusion_type="hidden",
            use_audio_placeholder=False,
        ),
        train=TrainConfig(
            epochs=10,
            batch_size=4,
            learning_rate=2e-5,
            save_dir="./outputs/baseline_hidden",
        ),
    )