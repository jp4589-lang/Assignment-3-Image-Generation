from typing import Tuple, Optional
import torch
from torch import nn

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Returns (avg_loss, accuracy_percent).
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval().to(device)
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(len(data_loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc
