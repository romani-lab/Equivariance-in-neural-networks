"""Loss functions and angle representations for orientation regression."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AngleRepresentation(ABC):
    """Abstract base class for target representations.

    Handles encoding angles to a representation suitable for regression,
    and decoding predictions back to angles.
    """

    output_dim: int  # Number of output dimensions for the representation
    ordered: bool  # Whether this representation distinguishes angle sign

    @abstractmethod
    def encode(self, delta_rad: torch.Tensor) -> torch.Tensor:
        """Encode angle difference to representation.

        Args:
            delta_rad: Angle difference (angle2 - angle1) in radians

        Returns:
            Encoded representation tensor
        """
        pass

    @abstractmethod
    def decode(self, pred: torch.Tensor) -> torch.Tensor:
        """Decode prediction to angle in degrees.

        Args:
            pred: Model prediction tensor

        Returns:
            Angle in degrees
        """
        pass


class SinCosOrdered(AngleRepresentation):
    """(sin, cos) representation for ordered image pairs.

    Encodes to (sin(delta), cos(delta)) where order matters.
    Decodes to angles in [0, 360).
    """

    output_dim = 2
    ordered = True

    def encode(self, delta_rad: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(delta_rad), torch.cos(delta_rad)], dim=1)

    def decode(self, pred: torch.Tensor) -> torch.Tensor:
        angle_rad = torch.atan2(pred[:, 0], pred[:, 1])
        angle_deg = angle_rad * 180.0 / torch.pi
        return angle_deg % 360


class SinCosUnordered(AngleRepresentation):
    """(|sin|, cos) representation for unordered image pairs.

    Encodes to (|sin(delta)|, cos(delta)) where order doesn't matter.
    Decodes to angles in [0, 180].
    """

    output_dim = 2
    ordered = False

    def encode(self, delta_rad: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.abs(torch.sin(delta_rad)), torch.cos(delta_rad)], dim=1)

    def decode(self, pred: torch.Tensor) -> torch.Tensor:
        angle_rad = torch.atan2(pred[:, 0], pred[:, 1])
        angle_deg = angle_rad * 180.0 / torch.pi
        return torch.abs(angle_deg)  # Map to [0, 180]


class CosOnly(AngleRepresentation):
    """Cosine-only representation.

    Cosine alone is sufficient to encode angles in [0, 180] since cos is
    injective on that range. Naturally unordered since cos(a) = cos(-a).
    """

    output_dim = 1
    ordered = False

    def encode(self, delta_rad: torch.Tensor) -> torch.Tensor:
        return torch.cos(delta_rad)

    def decode(self, pred: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        pred = torch.clamp(pred, -1.0, 1.0)
        angle_rad = torch.acos(pred)
        return angle_rad * 180.0 / torch.pi


def get_angle_representation(name: str) -> AngleRepresentation:
    """Get angle representation by name.

    Args:
        name: One of:
            - "sincos_ordered": (sin, cos) for ordered pairs, decodes to [0, 360)
            - "sincos_unordered": (|sin|, cos) for unordered pairs, decodes to [0, 180]
            - "cos": cosine only (unordered), decodes to [0, 180]

    Returns:
        AngleRepresentation instance
    """
    representations = {
        "sincos_ordered": SinCosOrdered,
        "sincos_unordered": SinCosUnordered,
        "cos": CosOnly,
    }
    if name not in representations:
        msg = f"Unknown representation: {name}. Choose from {list(representations.keys())}"
        raise ValueError(msg)
    return representations[name]()


class OrientationLoss(nn.Module):
    """MSE loss on angle representation."""

    def __init__(self, representation: AngleRepresentation):
        super().__init__()
        self.representation = representation
        self.mse = nn.MSELoss()

    @property
    def ordered(self) -> bool:
        return self.representation.ordered

    @property
    def output_dim(self) -> int:
        return self.representation.output_dim

    def forward(self, pred: torch.Tensor, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between prediction and encoded target.

        Args:
            pred: Model prediction
            angle1: First rotation angle in degrees, shape (batch_size,)
            angle2: Second rotation angle in degrees, shape (batch_size,)

        Returns:
            MSE loss
        """
        # Handle (batch,) vs (batch, 1) for cos representation
        if self.representation.output_dim == 1:
            pred = pred.squeeze(-1) if pred.dim() > 1 else pred

        delta_rad = (angle2 - angle1) * torch.pi / 180.0
        target = self.representation.encode(delta_rad)
        return self.mse(pred, target)


class AngularLoss(nn.Module):
    """Cosine-based angular loss that directly penalizes angle error.

    Instead of MSE on (sin, cos) representation, this loss:
    1. Decodes the prediction to an angle
    2. Computes the angular difference to the target
    3. Returns 1 - cos(difference), which is 0 for perfect match and 2 for 180° error

    This avoids the regression-to-mean problem where MSE encourages predicting
    the expected value of the target distribution.
    """

    def __init__(self, representation: AngleRepresentation):
        super().__init__()
        self.representation = representation

    @property
    def ordered(self) -> bool:
        return self.representation.ordered

    @property
    def output_dim(self) -> int:
        return self.representation.output_dim

    def forward(self, pred: torch.Tensor, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """Compute angular loss between prediction and target.

        Args:
            pred: Model prediction (sin/cos values or cos value)
            angle1: First rotation angle in degrees, shape (batch_size,)
            angle2: Second rotation angle in degrees, shape (batch_size,)

        Returns:
            Mean angular loss in range [0, 2]
        """
        # Decode prediction to angle in degrees
        pred_deg = self.representation.decode(pred)

        # Compute target angle
        if self.representation.ordered:
            # Ordered: target is (angle2 - angle1) mod 360
            target_deg = (angle2 - angle1) % 360
        else:
            # Unordered: target is |angle2 - angle1| wrapped to [0, 180]
            delta = torch.abs(angle2 - angle1)
            target_deg = torch.min(delta, 360 - delta)

        # Compute angular difference in radians
        diff_rad = (pred_deg - target_deg) * torch.pi / 180.0

        # Loss = 1 - cos(diff), ranges from 0 (perfect) to 2 (180° error)
        loss = 1 - torch.cos(diff_rad)

        return loss.mean()


def get_orientation_loss(representation: str, loss_type: str = "mse") -> OrientationLoss | AngularLoss:
    """Get orientation loss by name.

    Args:
        representation: Angle representation name (see get_angle_representation)
        loss_type: "mse" for MSE on representation, "cosine" for cosine-based angular loss

    Returns:
        OrientationLoss or AngularLoss with the specified representation
    """
    rep = get_angle_representation(representation)
    if loss_type == "mse":
        return OrientationLoss(rep)
    elif loss_type == "cosine":
        return AngularLoss(rep)
    else:
        msg = f"Unknown loss_type: {loss_type}. Choose from ['mse', 'cosine']"
        raise ValueError(msg)
