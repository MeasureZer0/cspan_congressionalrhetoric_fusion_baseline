import torch
from torch import nn


class Projector(nn.Module):
    """Linear → LayerNorm → ReLU projection."""

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
    """
    Attend across modality tokens then classify.

    Input:  list of M tensors each shaped [B, common_dim]
    Output: logits [B, num_classes]
    """

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
        self.common_dim = common_dim

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

    def _attend(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Stack → add type embeddings → self-attention → norm."""
        B = embeddings[0].size(0)
        device = embeddings[0].device
        x = torch.stack(embeddings, dim=1)  # [B, M, D]
        type_ids = torch.arange(self.num_modalities, device=device)
        x = x + self.type_embeddings(type_ids).unsqueeze(0)
        attended, _ = self.attn(x, x, x)
        x = self.norm(x + attended)
        return x.reshape(B, -1)  # [B, M*D]

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self._attend(embeddings))

    def forward_hidden(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        return self._attend(embeddings)


class MultimodalFusionModel(nn.Module):
    """
    Wraps three encoders + three projection heads + one fusion module.

    Supports:
        forward(...)                   all three modalities (multimodal)
        forward_unimodal(modality, …)  single modality + masked others
        forward_hidden(...)            pre-classifier representation
    """

    MODALITIES = ("video", "text", "audio")

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
        self.num_classes = fusion.classifier[-1].out_features

        self.absent_video = nn.Parameter(torch.zeros(1, common_dim))
        self.absent_text = nn.Parameter(torch.zeros(1, common_dim))
        self.absent_audio = nn.Parameter(torch.zeros(1, common_dim))

    def _video_embed(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        if not any(p.requires_grad for p in self.video_encoder.parameters()):
            with torch.no_grad():
                h = self.video_encoder.forward_hidden(faces, pose, lengths)
        else:
            h = self.video_encoder.forward_hidden(faces, pose, lengths)
        return self.video_proj(h)

    def _text_embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if not any(p.requires_grad for p in self.text_encoder.parameters()):
            with torch.no_grad():
                h = self.text_encoder.forward_hidden(input_ids, attention_mask)
        else:
            h = self.text_encoder.forward_hidden(input_ids, attention_mask)
        return self.text_proj(h)

    def _audio_embed(self, waveform: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if not any(p.requires_grad for p in self.audio_encoder.parameters()):
            with torch.no_grad():
                h = self.audio_encoder.forward_hidden(waveform, attention_mask)
        else:
            h = self.audio_encoder.forward_hidden(waveform, attention_mask)
        return self.audio_proj(h)

    def _absent(self, token: nn.Parameter, B: int) -> torch.Tensor:
        """Expand an absent-modality token to batch size B."""
        return token.expand(B, -1)

    def forward(
        self,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        waveform: torch.Tensor,
        audio_attention_mask=None,
    ) -> dict:
        """Full multimodal forward pass."""
        v_emb = self._video_embed(faces, pose, lengths)
        t_emb = self._text_embed(input_ids, attention_mask)
        a_emb = self._audio_embed(waveform, audio_attention_mask)
        return {"logits": self.fusion([v_emb, t_emb, a_emb])}

    def forward_hidden(
        self,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        waveform: torch.Tensor,
        audio_attention_mask=None,
    ) -> torch.Tensor:
        """Pre-classifier fused representation (multimodal)."""
        v_emb = self._video_embed(faces, pose, lengths)
        t_emb = self._text_embed(input_ids, attention_mask)
        a_emb = self._audio_embed(waveform, audio_attention_mask)
        return self.fusion.forward_hidden([v_emb, t_emb, a_emb])

    def forward_unimodal(
        self,
        modality: str,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        waveform: torch.Tensor,
        audio_attention_mask=None,
    ) -> dict:
        """
        Single-modality forward pass.

        The active modality is encoded normally; the other two slots receive
        the learned absent-modality tokens so the fusion module always sees
        three inputs.

        Args:
            modality: one of 'video', 'text', 'audio'
        """
        assert modality in self.MODALITIES, (
            f"modality must be one of {self.MODALITIES}, got '{modality}'"
        )

        B = labels_batch_size(faces, input_ids, waveform)

        if modality == "video":
            v_emb = self._video_embed(faces, pose, lengths)
            t_emb = self._absent(self.absent_text, B)
            a_emb = self._absent(self.absent_audio, B)
        elif modality == "text":
            v_emb = self._absent(self.absent_video, B)
            t_emb = self._text_embed(input_ids, attention_mask)
            a_emb = self._absent(self.absent_audio, B)
        else:  # audio
            v_emb = self._absent(self.absent_video, B)
            t_emb = self._absent(self.absent_text, B)
            a_emb = self._audio_embed(waveform, audio_attention_mask)

        return {"logits": self.fusion([v_emb, t_emb, a_emb])}

    def get_modality_embeddings(
        self,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        waveform: torch.Tensor,
        audio_attention_mask=None,
    ) -> dict:
        """
        Return the projected embedding for each modality separately.
        Useful for analysis and visualisation.
        """
        v_emb = self._video_embed(faces, pose, lengths)
        t_emb = self._text_embed(input_ids, attention_mask)
        a_emb = self._audio_embed(waveform, audio_attention_mask)
        return {"video": v_emb, "text": t_emb, "audio": a_emb}


def labels_batch_size(*tensors) -> int:
    """Return batch size from the first non-None tensor in the list."""
    for t in tensors:
        if t is not None:
            return t.size(0)
    raise ValueError("All tensors are None – cannot determine batch size.")
