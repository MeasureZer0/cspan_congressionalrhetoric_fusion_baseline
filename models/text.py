import torch
import torch.nn as nn
from transformers import BertModel


class BertTextClassifier(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze: bool = False,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        hidden_size = self.bert.config.hidden_size

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.output_dim = hidden_size
        self.num_classes = num_classes

    def forward_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # CLS token
        return self.dropout(cls)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids, attention_mask)
        return self.classifier(hidden)
