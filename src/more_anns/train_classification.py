"""Training script for AlexNet classification (Task 1).

Based on PyTorch Vision training recipe:
https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg
"""

import logging

import torch
from torch import nn
from tqdm import tqdm

from more_anns.data import ImageNetDataModule
from more_anns.metrics import accuracy
from more_anns.models import AlexNet
from more_anns.train_base import BaseTrainer

log = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    """Trainer for AlexNet classification task."""

    def __init__(
        self,
        dataset: str,
        output_dir: str,
        *,
        use_circular_mask: bool = True,
        use_rotation_aug: bool = False,
        rotation_quantization: int | None = None,
        interpolation_order: int = 1,
        num_epochs: int = 90,
        batch_size: int = 256,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        num_workers: int = 4,
        device: str = "cuda",
        pretrained_weights: str | None = None,
        pretrained_layers: str | None = None,
        freeze_backbone: bool = False,
        unfreeze_at_epoch: int | None = None,
        hf_repo_id: str | None = None,
        backbone: str = "alexnet",
    ):
        super().__init__(
            output_dir=output_dir,
            num_epochs=num_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            freeze_backbone=freeze_backbone,
            unfreeze_at_epoch=unfreeze_at_epoch,
            pretrained_layers=pretrained_layers,
            hf_repo_id=hf_repo_id,
        )

        self.data_module = ImageNetDataModule(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            return_pair=False,
            return_label=True,
            return_angles=False,
            use_circular_mask=use_circular_mask,
            use_rotation_aug=use_rotation_aug,
            rotation_quantization=rotation_quantization,
            interpolation_order=interpolation_order,
        )

        # Backbone: default AlexNet, or an escnn CNN with built-in rotation equivariance (SI Fig 6).
        # Equivariant backbones carry their own group-pooled FC head, so AlexNet's task head
        # is not used; the trainer only needs (logits,) from forward, which both provide.
        if backbone.startswith("eqv_c"):
            from more_anns.equivariant_models import build_equivariant_alexnet
            self.model = build_equivariant_alexnet(backbone, num_classes=self.data_module.NUM_CLASSES)
        elif backbone == "alexnet":
            self.model = AlexNet(
                task="classification",
                num_classes=self.data_module.NUM_CLASSES,
                pretrained_weights=pretrained_weights,
                pretrained_layers=pretrained_layers,
            )
        else:
            raise ValueError(f"unknown backbone {backbone!r} (expected 'alexnet' or an equivariant_models.BACKBONES key: 'eqv_c15', 'eqv_c15_restricted', 'eqv_c1')")

        self.criterion = nn.CrossEntropyLoss()

        self.config = {
            "dataset": dataset,
            "backbone": backbone,
            "num_classes": self.data_module.NUM_CLASSES,
            "use_circular_mask": use_circular_mask,
            "use_rotation_aug": use_rotation_aug,
            "rotation_quantization": rotation_quantization,
            "interpolation_order": interpolation_order,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "device": device,
            "pretrained_weights": pretrained_weights,
            "pretrained_layers": pretrained_layers,
            "freeze_backbone": freeze_backbone,
            "unfreeze_at_epoch": unfreeze_at_epoch,
        }

        self._setup_training()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        train_loader = self.data_module.train_dataloader()

        metrics = {"loss": 0.0, "top1": 0.0, "top5": 0.0}
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)  # noqa: PLW2901
            labels = labels.to(self.device)  # noqa: PLW2901

            (outputs,) = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            metrics["loss"] += loss.item() * batch_size
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            metrics["top1"] += top1 * batch_size
            metrics["top5"] += top5 * batch_size
            total += batch_size

            pbar.set_postfix({"loss": loss.item(), "top1": metrics["top1"] / total, "top5": metrics["top5"] / total})

        for k in metrics:
            metrics[k] /= total
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        val_loader = self.data_module.val_dataloader()

        metrics = {"loss": 0.0, "top1": 0.0, "top5": 0.0}
        total = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")
        for images, labels in pbar:
            images = images.to(self.device)  # noqa: PLW2901
            labels = labels.to(self.device)  # noqa: PLW2901

            (outputs,) = self.model(images)
            loss = self.criterion(outputs, labels)

            batch_size = images.size(0)
            metrics["loss"] += loss.item() * batch_size
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            metrics["top1"] += top1 * batch_size
            metrics["top5"] += top5 * batch_size
            total += batch_size

            pbar.set_postfix({"loss": loss.item(), "top1": metrics["top1"] / total, "top5": metrics["top5"] / total})

        for k in metrics:
            metrics[k] /= total
        return metrics

    def _is_best(self, val_metrics, best):
        top1 = val_metrics["top1"]
        is_better = best is None or top1 > best
        new_best = top1 if is_better else best
        return is_better, new_best, f"Top-1: {new_best:.2f}%"

    def _log_metrics(self, train_metrics, val_metrics, epoch):
        log.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Top-1: {train_metrics['top1']:.2f}%, Top-5: {train_metrics['top5']:.2f}%"
        )
        log.info(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Top-1: {val_metrics['top1']:.2f}%, Top-5: {val_metrics['top5']:.2f}%"
        )

        self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        self.writer.add_scalar("Accuracy_Top1/train", train_metrics["top1"], epoch)
        self.writer.add_scalar("Accuracy_Top1/val", val_metrics["top1"], epoch)
        self.writer.add_scalar("Accuracy_Top5/train", train_metrics["top5"], epoch)
        self.writer.add_scalar("Accuracy_Top5/val", val_metrics["top5"], epoch)
