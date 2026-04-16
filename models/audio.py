from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForAudioClassification

TARGET_SR = 16_000


class ChunkedWav2VecAudioEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "superb/wav2vec2-large-superb-er",
        num_classes: int = 3,
        dropout: float = 0.3,
        chunk_secs: int = 5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.chunk_secs = chunk_secs
        self.chunk_len = chunk_secs * TARGET_SR
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.output_dim = self.hidden_size
        self.num_classes = num_classes

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def _normalize_input_shape(self, audio_values: torch.Tensor) -> torch.Tensor:
        if audio_values.dim() == 3:
            if audio_values.size(1) != 1:
                raise ValueError(
                    f"Expected mono audio [B, 1, T], got {tuple(audio_values.shape)}"
                )
            audio_values = audio_values.squeeze(1)
        elif audio_values.dim() != 2:
            raise ValueError(
                f"Expected [B, T] or [B, 1, T], got {tuple(audio_values.shape)}"
            )
        return audio_values

    def _chunk_single(self, waveform: torch.Tensor) -> torch.Tensor:
        total_len = waveform.size(0)
        if total_len == 0:
            return waveform.new_zeros(1, self.chunk_len)
        num_chunks = (total_len + self.chunk_len - 1) // self.chunk_len
        padded_len = num_chunks * self.chunk_len
        if padded_len > total_len:
            waveform = F.pad(waveform, (0, padded_len - total_len))
        return waveform.view(num_chunks, self.chunk_len)

    def _encode_chunk_batch(self, chunk_batch: torch.Tensor) -> torch.Tensor:
        ctx = (
            torch.no_grad()
            if not any(p.requires_grad for p in self.model.parameters())
            else torch.enable_grad()
        )
        with ctx:
            outputs = self.model(
                input_values=chunk_batch,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1].mean(dim=1)  # [num_chunks, H]

    def forward_hidden(
        self, audio_values: torch.Tensor, attention_mask=None
    ) -> torch.Tensor:
        audio_values = self._normalize_input_shape(audio_values)
        batch_embeddings: List[torch.Tensor] = []
        for i in range(audio_values.size(0)):
            chunks = self._chunk_single(audio_values[i])
            chunk_embeddings = self._encode_chunk_batch(chunks)
            batch_embeddings.append(chunk_embeddings.mean(dim=0))
        pooled = torch.stack(batch_embeddings, dim=0)  # [B, H]
        return self.dropout(pooled)

    def forward(self, audio_values: torch.Tensor, attention_mask=None) -> torch.Tensor:
        return self.classifier(self.forward_hidden(audio_values))
