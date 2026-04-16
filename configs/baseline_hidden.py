from config_base import *


def get_config():
    return FullConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(),
        train=TrainConfig(
            epochs=30,
            batch_size=2,
            learning_rate=2e-5,
            save_dir="./outputs/baseline_hidden",
        ),
    )
