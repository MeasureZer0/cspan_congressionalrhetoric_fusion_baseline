# Multimodal Congressional Speech Analysis

This project implements baseline **multimodal fusion models** for analyzing congressional speeches using three modalities:

* **Text** (BERT)
* **Audio** (Wav2Vec2)
* **Video** (face + pose encoder)

The system supports multiple fusion strategies and experiment configurations designed for reproducible research.

---

# Project Overview

The goal is to classify congressional speech segments using information from:

* the **transcript**
* the **speaker's voice**
* the **speaker's facial expressions and body pose**

The system implements two core multimodal fusion baselines:

### 1️⃣ Late Fusion

Each modality produces its own classification logits.

These logits are concatenated and passed to a small fusion classifier.

```
Text → classifier → logits
Audio → classifier → logits
Video → classifier → logits

[logits_text | logits_audio | logits_video]
                ↓
           fusion MLP
                ↓
           final prediction
```

### 2️⃣ Hidden Fusion

Each modality produces a latent representation.

These representations are concatenated and classified jointly.

```
Text → embedding
Audio → embedding
Video → embedding

[hidden_text | hidden_audio | hidden_video]
                    ↓
               fusion MLP
                    ↓
             final prediction
```

Hidden fusion typically performs better but late fusion is easier to interpret.

---

# Repository Structure

```
multimodal_fusion/
│
├── main.py
├── trainer.py
│
├── config_base.py
├── configs/
│
├── datasets/
│
├── models/
│
├── utils/
│
└── outputs/
```

Detailed explanation below.

---

# Main Training Entry Point

### `main.py`

The main script responsible for:

* loading experiment configuration
* building datasets
* building models
* launching training

Example usage:

```bash
python main.py --config configs.baseline_hidden
```

---

# Training Logic

### `train.py`

Handles the full training pipeline:

* training loop
* validation loop
* metric computation
* checkpoint saving
* experiment logging

Metrics tracked:

* accuracy
* macro F1
* confusion matrix

---

# Configuration System

Experiment parameters are controlled through **Python config files**.

This makes experiments easy to reproduce and version-control.

## Base Config

### `config_base.py`

Defines the configuration dataclasses:

```
DatasetConfig
ModelConfig
TrainConfig
FullConfig
```

These contain parameters for:

* dataset locations
* model architecture
* training hyperparameters

---

## Experiment Configurations

Located in:

```
configs/
```

Each file defines **one experiment setup**.

Example configs:

```
configs/
├── baseline_hidden.py
└── baseline_late.py
```

Each config exposes:

```python
def get_config():
    return FullConfig(...)
```

### Example

```
configs/baseline_hidden.py
```

Defines the default multimodal baseline:

* hidden fusion

---

# Dataset

Dataset loading logic is located in:

```
datasets/
```

### `multimodal_classification.py`

Loads multimodal samples containing:

```
{
  "text": transcript tokens
  "audio": waveform
  "video": frame features
  "label": class label
}
```

---

# Models

Located in:

```
models/
```

## Text Encoder

```
models/text.py
```

Uses:

```
BertModel
```

The model outputs:

* CLS embedding (hidden fusion)
* classification logits (late fusion)

---

## Audio Encoder

```
models/audio.py
```

### Wav2Vec2 Encoder

```
Wav2Vec2Model
```

---

## Video Encoder Adapter

```
models/video.py
```

Wraps the video model to ensure consistent interface:

Required methods:

```
forward_hidden()
forward()
```

Required attributes:

```
output_dim
num_classes
```

---

## Fusion Model

```
models/fuse.py
```

### CrossModalAttentionFusion

Attention-based fusion over projected modality embeddings.

---

# Utilities

Located in:

```
utils/
```

### `collate.py`

Custom DataLoader collate function.

Handles:

* text batching
* audio padding
* variable-length video sequences

---

### `metrics.py`

Implements:

* confusion matrix
* accuracy
* macro F1 score

---

# Running Experiments

Run experiments by selecting a config file.

Example:

### Attention Fusion Baseline

```
python main.py --config configs.baseline_hidden
```

# Outputs

Training results are stored in:

```
outputs/
```

Each run saves:

```
outputs/run_name/
├── best.pt
├── history.json
└── config.json
```

Where:

* `best.pt` = best model checkpoint
* `history.json` = training metrics
* `config.json` = full experiment configuration

---

# Recommended Experiments

For a solid baseline comparison:

| Model         | Description        |
| ------------- | ------------------ |
| text_only     | BERT classifier    |
| video_only    | video encoder      |
| audio_only    | wav2vec classifier |
| late_fusion   | logits fusion      |
| hidden_fusion | latent fusion      |

These provide a full multimodal ablation study.
