from typing import Optional

import torch
from torch import nn


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttentionFusion(nn.Module):
    def __init__(
        self,
        common_dim: int = 256,
        num_modalities: int = 3,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.type_embeddings = nn.Embedding(num_modalities, common_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(common_dim)
        fused_dim = common_dim * num_modalities
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )
        self.output_dim = fused_dim // 2

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        B = embeddings[0].size(0)
        device = embeddings[0].device
        x = torch.stack(embeddings, dim=1)
        type_ids = torch.arange(self.num_modalities, device=device)
        x = x + self.type_embeddings(type_ids).unsqueeze(0)
        attended, _ = self.attn(x, x, x)
        x = self.norm(x + attended)
        x = x.reshape(B, -1)
        return self.classifier(x)

    def forward_hidden(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        B = embeddings[0].size(0)
        device = embeddings[0].device
        x = torch.stack(embeddings, dim=1)
        type_ids = torch.arange(self.num_modalities, device=device)
        x = x + self.type_embeddings(type_ids).unsqueeze(0)
        attended, _ = self.attn(x, x, x)
        x = self.norm(x + attended)
        return x.reshape(B, -1)


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        video_encoder: nn.Module,
        video_dim: int,
        text_encoder: nn.Module,
        text_dim: int,
        audio_encoder: nn.Module,
        audio_dim: int,
        fusion: nn.Module,
        common_dim: int = 256,
    ) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

        self.video_proj = Projector(video_dim, common_dim)
        self.text_proj = Projector(text_dim, common_dim)
        self.audio_proj = Projector(audio_dim, common_dim)

        self.fusion = fusion
        self.common_dim = common_dim

    def _video_embed(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.video_proj(self.video_encoder.forward_hidden(faces, pose, lengths))

    def _text_embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.text_proj(self.text_encoder(input_ids, attention_mask))

    def _audio_embed(self, audio_values: torch.Tensor) -> torch.Tensor:
        return self.audio_proj(self.audio_encoder(audio_values))

    def forward(
        self,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_values: torch.Tensor,
    ) -> torch.Tensor:
        v_emb = self._video_embed(faces, pose, lengths)
        t_emb = self._text_embed(input_ids, attention_mask)
        a_emb = self._audio_embed(audio_values)
        return self.fusion([v_emb, t_emb, a_emb])

    def forward_hidden(
        self,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_values: torch.Tensor,
    ) -> torch.Tensor:
        v_emb = self._video_embed(faces, pose, lengths)
        t_emb = self._text_embed(input_ids, attention_mask)
        a_emb = self._audio_embed(audio_values)
        return self.fusion.forward_hidden([v_emb, t_emb, a_emb])