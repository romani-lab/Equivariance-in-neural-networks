"""Base trainer with shared optimizer, scheduler, checkpointing, and training loop."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import HFSummaryWriter
from torch import optim
from torch.utils.tensorboard import SummaryWriter

mpl.use("Agg")

# HFSummaryWriter bug: passes logdir= to SummaryWriter which only accepts log_dir=.
# Patch SummaryWriter.__init__ to accept both.
_orig_sw_init = SummaryWriter.__init__


def _sw_init_compat(self, logdir=None, log_dir=None, **kwargs):
    _orig_sw_init(self, log_dir=log_dir or logdir, **kwargs)
    self.logdir = self.log_dir


SummaryWriter.__init__ = _sw_init_compat

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base trainer with shared optimizer, scheduler, checkpointing, and training loop.

    Subclasses must:
    1. Call super().__init__() with common training params
    2. Set self.model and self.config
    3. Call self._setup_training() to create optimizer/scheduler and freeze backbone
    """

    def __init__(
        self,
        output_dir: str | Path,
        num_epochs: int = 90,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        *,
        freeze_backbone: bool = False,
        unfreeze_at_epoch: int | None = None,
        pretrained_layers: str | None = None,
        hf_repo_id: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.device = device
        self._freeze_backbone = freeze_backbone
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._pretrained_layers = pretrained_layers
        self._optim_kwargs = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        log_dir = self.output_dir / "logs"
        if hf_repo_id:
            # Use the last 2 path parts as path_in_repo so runs don't overwrite
            # each other (output_dir is like .../more_anns/<experiment>).
            path_in_repo = "/".join(self.output_dir.parts[-2:])
            self.writer = HFSummaryWriter(
                repo_id=hf_repo_id, logdir=str(log_dir), path_in_repo=path_in_repo
            )
        else:
            self.writer = SummaryWriter(log_dir=log_dir)

    def _setup_training(self) -> None:
        """Move model to device, freeze backbone, create optimizer and scheduler.

        Call this at the end of subclass __init__ after self.model is set.
        """
        self.model = self.model.to(self.device)
        if self._freeze_backbone and hasattr(self.model, "backbone"):
            for name, param in self.model.backbone.named_parameters():
                if self._pretrained_layers is None or name.startswith(self._pretrained_layers):
                    param.requires_grad = False
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self._optim_kwargs,
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def _maybe_unfreeze(self, epoch: int) -> None:
        """Unfreeze backbone and rebuild optimizer/scheduler at the configured epoch."""
        if not self._freeze_backbone or epoch != self._unfreeze_at_epoch:
            return
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        self._freeze_backbone = False
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer = optim.SGD(
            self.model.parameters(),
            **{**self._optim_kwargs, "lr": current_lr},
        )
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", self._optim_kwargs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1,
            last_epoch=epoch - 2,
        )
        log.info(f"Unfreezing backbone at epoch {epoch} (lr={current_lr})")

    def save_checkpoint(self, epoch: int, *, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            log.info(f"Saved best model to {best_path}")

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, float]: ...

    @abstractmethod
    def validate(self, epoch: int) -> dict[str, float]: ...

    @abstractmethod
    def _is_best(self, val_metrics: dict[str, float], best: float | None) -> tuple[bool, float, str]:
        """Check if current val metrics beat the best so far.

        Returns:
            (is_best, new_best_value, label) where label is e.g. "Top-1: 95.2%"
        """
        ...

    @abstractmethod
    def _log_metrics(self, train_metrics: dict[str, float], val_metrics: dict[str, float], epoch: int) -> None:
        """Print epoch summary and log scalars to tensorboard."""
        ...

    def _log_angle_heatmap(
        self,
        all_pred: torch.Tensor,
        all_target: torch.Tensor,
        *,
        ordered: bool,
        epoch: int,
    ) -> None:
        """Log 2D heatmap of predicted vs target angles to TensorBoard.

        Creates a heatmap where each column (target angle bin) shows the
        distribution of predictions, normalized so each column sums to 1.
        A perfect model produces a bright diagonal.
        """
        max_angle = 360.0 if ordered else 180.0
        n_bins = 36 if ordered else 18
        bin_size = max_angle / n_bins

        pred_bins = (all_pred / bin_size).long().clamp(0, n_bins - 1)
        target_bins = (all_target / bin_size).long().clamp(0, n_bins - 1)

        indices = pred_bins * n_bins + target_bins
        heatmap = torch.bincount(indices, minlength=n_bins * n_bins).float().view(n_bins, n_bins)

        # Normalize per target bin (columns)
        col_sums = heatmap.sum(dim=0, keepdim=True).clamp(min=1)
        heatmap = heatmap / col_sums

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            heatmap.numpy(),
            origin="lower",
            aspect="auto",
            extent=[0, max_angle, 0, max_angle],
            vmin=0,
        )
        ax.plot([0, max_angle], [0, max_angle], "r--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Target angle")
        ax.set_ylabel("Predicted angle")
        fig.colorbar(im, ax=ax, label="P(pred | target)")
        fig.tight_layout()

        self.writer.add_figure("predictions/angle_heatmap", fig, epoch)
        plt.close(fig)

    def _log_angle_rose(
        self,
        all_pred: torch.Tensor,
        all_target: torch.Tensor,
        *,
        ordered: bool,
        epoch: int,
    ) -> None:
        """Log polar rose diagrams of predicted and target angle distributions."""
        max_angle = 360.0 if ordered else 180.0
        n_bins = 36 if ordered else 18

        bin_edges = np.linspace(0, max_angle, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        theta = np.radians(bin_centers)
        width = np.radians(max_angle / n_bins)

        pred_np = all_pred.numpy()
        target_np = all_target.numpy()
        pred_counts, _ = np.histogram(pred_np, bins=bin_edges)
        target_counts, _ = np.histogram(target_np, bins=bin_edges)
        pred_counts = pred_counts / max(pred_counts.sum(), 1)
        target_counts = target_counts / max(target_counts.sum(), 1)

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            subplot_kw={"projection": "polar"},
            figsize=(8, 4),
        )

        for ax, counts, title, color in [
            (ax1, target_counts, "Target angles", "#4A90D9"),
            (ax2, pred_counts, "Predicted angles", "#E24A33"),
        ]:
            ax.bar(
                theta,
                counts,
                width=width,
                bottom=0,
                color=color,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_yticklabels([])
            ax.set_title(title, fontsize=10, pad=15)
            if ordered:
                ax.set_xticks(np.arange(0, 360, 30) * np.pi / 180)
                ax.set_xticklabels([f"{a:.0f}" for a in np.arange(0, 360, 30)], fontsize=8)
            else:
                ax.set_thetamin(0)
                ax.set_thetamax(180)
                ax.set_xticks(np.arange(0, 181, 30) * np.pi / 180)
                ax.set_xticklabels([f"{a:.0f}" for a in np.arange(0, 181, 30)], fontsize=8)

        fig.tight_layout()
        self.writer.add_figure("predictions/angle_rose", fig, epoch)
        plt.close(fig)

    def train(self) -> None:
        """Run full training loop."""
        log.info(f"Starting training with config: {self.config}")
        best_metric = None
        best_label = ""

        for epoch in range(1, self.num_epochs + 1):
            self._maybe_unfreeze(epoch)

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self._log_metrics(train_metrics, val_metrics, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            self.scheduler.step()

            is_best, best_metric, best_label = self._is_best(val_metrics, best_metric)
            self.save_checkpoint(epoch, is_best=is_best)
            log.info(f"Best validation {best_label}\n")

        log.info(f"Training complete! Best {best_label}")
        self.writer.close()
