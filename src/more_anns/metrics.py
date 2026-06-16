"""Metrics for training and evaluation."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from more_anns.losses import AngleRepresentation


def orientation_mae(
    pred: torch.Tensor,
    angle1: torch.Tensor,
    angle2: torch.Tensor,
    representation: "AngleRepresentation",
    *,
    ordered: bool | None = None,
) -> torch.Tensor:
    """Compute mean absolute error using angle representation.

    Args:
        pred: Raw model prediction
        angle1: First rotation angle in degrees
        angle2: Second rotation angle in degrees
        representation: AngleRepresentation for decoding
        ordered: If None, use representation.ordered. Otherwise override.

    Returns:
        MAE in degrees, in [0, 180]
    """
    pred_angle = representation.decode(pred)
    use_ordered = representation.ordered if ordered is None else ordered

    if use_ordered:
        target_angle = (angle2 - angle1) % 360
        diff = torch.abs(pred_angle - target_angle)
        diff = torch.min(diff, 360 - diff)  # Wrap to [0, 180]
    else:
        delta = torch.abs(angle2 - angle1)
        target_angle = torch.min(delta, 360 - delta)  # Target in [0, 180]
        pred_angle = torch.min(pred_angle, 360 - pred_angle)  # Fold to [0, 180]
        diff = torch.abs(pred_angle - target_angle)

    return diff.mean()


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[float]:
    """Compute top-k accuracy for the specified values of k.

    Args:
        output: Model output logits of shape (batch_size, num_classes)
        target: Ground truth labels of shape (batch_size,)
        topk: Tuple of k values to compute accuracy for

    Returns:
        List of accuracy values (as percentages) for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item() / batch_size * 100.0)
        return res
