import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.3, freeze: bool = False):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.wav2vec.config.hidden_size

        if freeze:
            for p in self.wav2vec.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.output_dim = hidden_size
        self.num_classes = num_classes

    def _masked_mean_pool(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # x: [B, T', H]
        if attention_mask is None:
            return x.mean(dim=1)

        # Wav2Vec2 downsamples time internally, so raw audio mask doesn't match x length directly.
        # For a simple baseline, fall back to unmasked mean pooling after encoder.
        return x.mean(dim=1)

    def forward_hidden(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # waveform: [B, T] or [B, 1, T]
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        outputs = self.wav2vec(input_values=waveform, attention_mask=attention_mask)
        pooled = self._masked_mean_pool(outputs.last_hidden_state, attention_mask)
        return self.dropout(pooled)

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.forward_hidden(waveform, attention_mask)
        return self.classifier(hidden)