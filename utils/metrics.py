import torch


def confusion_matrix_torch(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for p, t in zip(preds.view(-1), targets.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def macro_f1_from_confusion_matrix(cm: torch.Tensor) -> float:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)

    return sum(f1s) / len(f1s)
