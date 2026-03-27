import torch
import torch.nn as nn


class LateFusionModel(nn.Module):
    """
    Preserves unimodal classification heads and fuses their logits.
    """
    def __init__(
        self,
        video_model: nn.Module,
        text_model: nn.Module,
        audio_model: nn.Module,
        num_classes: int = 3,
        fusion_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.video_model = video_model
        self.text_model = text_model
        self.audio_model = audio_model
        self.num_classes = num_classes

        self.fusion_head = nn.Sequential(
            nn.Linear(num_classes * 3, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(
        self,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        waveform: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ):
        video_logits = self.video_model(faces, pose, lengths)
        text_logits = self.text_model(input_ids, attention_mask)
        audio_logits = self.audio_model(waveform, audio_attention_mask)

        fused_input = torch.cat([video_logits, text_logits, audio_logits], dim=-1)
        fused_logits = self.fusion_head(fused_input)

        return {
            "logits": fused_logits,
            "video_logits": video_logits,
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }


class HiddenFusionModel(nn.Module):
    """
    Uses modality hidden states, concatenates them, and learns a new classifier.
    """
    def __init__(
        self,
        video_model: nn.Module,
        text_model: nn.Module,
        audio_model: nn.Module,
        num_classes: int = 3,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.video_model = video_model
        self.text_model = text_model
        self.audio_model = audio_model
        self.num_classes = num_classes

        concat_dim = (
            video_model.output_dim +
            text_model.output_dim +
            audio_model.output_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)
        self.output_dim = fusion_hidden

    def forward(
        self,
        faces: torch.Tensor,
        pose: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        waveform: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ):
        v = self.video_model.forward_hidden(faces, pose, lengths)
        t = self.text_model.forward_hidden(input_ids, attention_mask)
        a = self.audio_model.forward_hidden(waveform, audio_attention_mask)

        fused = torch.cat([v, t, a], dim=-1)
        hidden = self.fusion(fused)
        logits = self.classifier(hidden)

        return {
            "logits": logits,
            "video_hidden": v,
            "text_hidden": t,
            "audio_hidden": a,
            "fusion_hidden": hidden,
        }