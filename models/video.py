import torch
import torch.nn as nn


class VideoClassifierAdapter(nn.Module):

    def __init__(self, video_model: nn.Module, checkpoint_path=None, freeze=False):
        super().__init__()

        self.video_model = video_model

        if checkpoint_path is not None:
            print(f"Loading pretrained video model from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # handle either raw state_dict or training checkpoint
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.video_model.load_state_dict(state_dict, strict=False)

        if freeze:
            for p in self.video_model.parameters():
                p.requires_grad = False

        self.output_dim = video_model.output_dim
        self.num_classes = video_model.num_classes

    def forward_hidden(self, faces, pose, lengths):
        return self.video_model.forward_hidden(faces, pose, lengths)

    def forward(self, faces, pose, lengths):
        return self.video_model(faces, pose, lengths)