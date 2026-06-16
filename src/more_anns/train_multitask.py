"""Training script for multi-task AlexNet (classification + orientation, Task 3)."""

import logging
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from more_anns.data import ImageNetDataModule
from more_anns.losses import get_angle_representation, get_orientation_loss
from more_anns.metrics import accuracy, orientation_mae
from more_anns.models import AlexNet
from more_anns.train_base import BaseTrainer

log = logging.getLogger(__name__)


class MultiTaskTrainer(BaseTrainer):
    """Trainer for multi-task AlexNet (classification + orientation)."""

    def __init__(
        self,
        dataset: str,
        output_dir: str | Path,
        *,
        use_circular_mask: bool = True,
        classification_weight: float = 1.0,
        orientation_weight: float = 1.0,
        angle_representation: str = "sincos_unordered",
        orientation_activation: str = "linear",
        orientation_fusion: str = "concat",
        similarity_layer: str = "classifier.5",
        loss_type: str = "mse",
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

        self.classification_weight = classification_weight
        self.orientation_weight = orientation_weight

        # Data - get image pairs with labels and rotation angles
        self.data_module = ImageNetDataModule(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            use_circular_mask=use_circular_mask,
            return_pair=True,
            return_label=True,
            return_angles=True,
            rotation_quantization=rotation_quantization,
            interpolation_order=interpolation_order,
        )

        # Losses
        self.classification_criterion = nn.CrossEntropyLoss()
        representation = get_angle_representation(angle_representation)
        self.orientation_criterion = get_orientation_loss(angle_representation, loss_type)

        # Model
        self.model = AlexNet(
            task="multitask",
            num_classes=self.data_module.NUM_CLASSES,
            orientation_fusion=orientation_fusion,
            orientation_output_dim=representation.output_dim,
            orientation_activation=orientation_activation,
            similarity_layer=similarity_layer,
            pretrained_weights=pretrained_weights,
            pretrained_layers=pretrained_layers,
        )

        self.config = {
            "dataset": dataset,
            "use_circular_mask": use_circular_mask,
            "classification_weight": classification_weight,
            "orientation_weight": orientation_weight,
            "angle_representation": angle_representation,
            "orientation_activation": orientation_activation,
            "orientation_fusion": orientation_fusion,
            "similarity_layer": similarity_layer,
            "loss_type": loss_type,
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

        rep = self.orientation_criterion.representation
        metrics = {
            "loss": 0.0,
            "cls_loss": 0.0,
            "orient_loss": 0.0,
            "orient_mae_unordered": 0.0,
            "cls_top1": 0.0,
            "cls_top5": 0.0,
        }
        if rep.ordered:
            metrics["orient_mae_ordered"] = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        for img1, img2, labels, angle1, angle2 in pbar:
            img1 = img1.to(self.device)  # noqa: PLW2901
            img2 = img2.to(self.device)  # noqa: PLW2901
            labels = labels.to(self.device)  # noqa: PLW2901
            angle1 = angle1.to(self.device, dtype=torch.float32)  # noqa: PLW2901
            angle2 = angle2.to(self.device, dtype=torch.float32)  # noqa: PLW2901

            class_logits1, class_logits2, orient_pred = self.model(img1, img2)

            cls_loss = (
                self.classification_criterion(class_logits1, labels)
                + self.classification_criterion(class_logits2, labels)
            ) / 2
            orient_loss = self.orientation_criterion(orient_pred, angle1, angle2)
            total_loss_batch = self.classification_weight * cls_loss + self.orientation_weight * orient_loss

            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            batch_size = img1.size(0)
            metrics["loss"] += total_loss_batch.item() * batch_size
            metrics["cls_loss"] += cls_loss.item() * batch_size
            metrics["orient_loss"] += orient_loss.item() * batch_size

            with torch.no_grad():
                mae_unord = orientation_mae(orient_pred, angle1, angle2, rep, ordered=False).item()
                metrics["orient_mae_unordered"] += mae_unord * batch_size
                if rep.ordered:
                    mae_ord = orientation_mae(orient_pred, angle1, angle2, rep, ordered=True).item()
                    metrics["orient_mae_ordered"] += mae_ord * batch_size

            top1_1, top5_1 = accuracy(class_logits1, labels, topk=(1, 5))
            top1_2, top5_2 = accuracy(class_logits2, labels, topk=(1, 5))
            metrics["cls_top1"] += (top1_1 + top1_2) / 2 * batch_size
            metrics["cls_top5"] += (top5_1 + top5_2) / 2 * batch_size
            total += batch_size

            pbar.set_postfix(
                {
                    "loss": total_loss_batch.item(),
                    "top1": metrics["cls_top1"] / total,
                    "mae": metrics["orient_mae_unordered"] / total,
                }
            )

        for k in metrics:
            metrics[k] /= total
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        val_loader = self.data_module.val_dataloader()

        rep = self.orientation_criterion.representation
        metrics = {
            "loss": 0.0,
            "cls_loss": 0.0,
            "orient_loss": 0.0,
            "orient_mae_unordered": 0.0,
            "cls_top1": 0.0,
            "cls_top5": 0.0,
        }
        if rep.ordered:
            metrics["orient_mae_ordered"] = 0.0
        total = 0
        all_pred_angles = []
        all_target_angles = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")
        for img1, img2, labels, angle1, angle2 in pbar:
            img1 = img1.to(self.device)  # noqa: PLW2901
            img2 = img2.to(self.device)  # noqa: PLW2901
            labels = labels.to(self.device)  # noqa: PLW2901
            angle1 = angle1.to(self.device, dtype=torch.float32)  # noqa: PLW2901
            angle2 = angle2.to(self.device, dtype=torch.float32)  # noqa: PLW2901

            class_logits1, class_logits2, orient_pred = self.model(img1, img2)

            cls_loss = (
                self.classification_criterion(class_logits1, labels)
                + self.classification_criterion(class_logits2, labels)
            ) / 2
            orient_loss = self.orientation_criterion(orient_pred, angle1, angle2)
            total_loss_batch = self.classification_weight * cls_loss + self.orientation_weight * orient_loss

            # Collect decoded angles for distribution analysis
            pred_deg = rep.decode(orient_pred)
            if rep.ordered:
                target_deg = (angle2 - angle1) % 360
            else:
                delta = torch.abs(angle2 - angle1)
                target_deg = torch.min(delta, 360 - delta)
            all_pred_angles.append(pred_deg.cpu())
            all_target_angles.append(target_deg.cpu())

            batch_size = img1.size(0)
            metrics["loss"] += total_loss_batch.item() * batch_size
            metrics["cls_loss"] += cls_loss.item() * batch_size
            metrics["orient_loss"] += orient_loss.item() * batch_size

            mae_unord = orientation_mae(orient_pred, angle1, angle2, rep, ordered=False).item()
            metrics["orient_mae_unordered"] += mae_unord * batch_size
            if rep.ordered:
                mae_ord = orientation_mae(orient_pred, angle1, angle2, rep, ordered=True).item()
                metrics["orient_mae_ordered"] += mae_ord * batch_size

            top1_1, top5_1 = accuracy(class_logits1, labels, topk=(1, 5))
            top1_2, top5_2 = accuracy(class_logits2, labels, topk=(1, 5))
            metrics["cls_top1"] += (top1_1 + top1_2) / 2 * batch_size
            metrics["cls_top5"] += (top5_1 + top5_2) / 2 * batch_size
            total += batch_size

            pbar.set_postfix(
                {
                    "loss": total_loss_batch.item(),
                    "top1": metrics["cls_top1"] / total,
                    "mae": metrics["orient_mae_unordered"] / total,
                }
            )

        for k in metrics:
            metrics[k] /= total

        # Log prediction distribution
        all_pred = torch.cat(all_pred_angles)
        all_target = torch.cat(all_target_angles)
        self.writer.add_histogram("predictions/angle", all_pred, epoch)
        self.writer.add_histogram("targets/angle", all_target, epoch)
        self._log_angle_heatmap(all_pred, all_target, ordered=rep.ordered, epoch=epoch)
        self._log_angle_rose(all_pred, all_target, ordered=rep.ordered, epoch=epoch)
        metrics["pred_mean"] = all_pred.mean().item()
        metrics["pred_std"] = all_pred.std().item()
        metrics["pred_min"] = all_pred.min().item()
        metrics["pred_max"] = all_pred.max().item()

        return metrics

    def _is_best(self, val_metrics, best):
        top1 = val_metrics["cls_top1"]
        is_better = best is None or top1 > best
        new_best = top1 if is_better else best
        return is_better, new_best, f"Top-1: {new_best:.2f}%"

    def _log_metrics(self, train_metrics, val_metrics, epoch):
        rep = self.orientation_criterion.representation

        msg = (
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Cls Loss: {train_metrics['cls_loss']:.4f}, "
            f"Orient Loss: {train_metrics['orient_loss']:.4f}, "
            f"Orient MAE: {train_metrics['orient_mae_unordered']:.2f}°"
        )
        if rep.ordered:
            msg += f", Orient MAE (ordered): {train_metrics['orient_mae_ordered']:.2f}°"
        msg += f", Top-1: {train_metrics['cls_top1']:.2f}%, Top-5: {train_metrics['cls_top5']:.2f}%"
        log.info(msg)

        msg = (
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Cls Loss: {val_metrics['cls_loss']:.4f}, "
            f"Orient Loss: {val_metrics['orient_loss']:.4f}, "
            f"Orient MAE: {val_metrics['orient_mae_unordered']:.2f}°"
        )
        if rep.ordered:
            msg += f", Orient MAE (ordered): {val_metrics['orient_mae_ordered']:.2f}°"
        msg += f", Top-1: {val_metrics['cls_top1']:.2f}%, Top-5: {val_metrics['cls_top5']:.2f}%"
        log.info(msg)

        self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/train_cls", train_metrics["cls_loss"], epoch)
        self.writer.add_scalar("Loss/val_cls", val_metrics["cls_loss"], epoch)
        self.writer.add_scalar("Loss/train_orient", train_metrics["orient_loss"], epoch)
        self.writer.add_scalar("Loss/val_orient", val_metrics["orient_loss"], epoch)
        self.writer.add_scalar("MAE/train", train_metrics["orient_mae_unordered"], epoch)
        self.writer.add_scalar("MAE/val", val_metrics["orient_mae_unordered"], epoch)
        if rep.ordered:
            self.writer.add_scalar("MAE_ordered/train", train_metrics["orient_mae_ordered"], epoch)
            self.writer.add_scalar("MAE_ordered/val", val_metrics["orient_mae_ordered"], epoch)
        self.writer.add_scalar("Accuracy_Top1/train", train_metrics["cls_top1"], epoch)
        self.writer.add_scalar("Accuracy_Top1/val", val_metrics["cls_top1"], epoch)
        self.writer.add_scalar("Accuracy_Top5/train", train_metrics["cls_top5"], epoch)
        self.writer.add_scalar("Accuracy_Top5/val", val_metrics["cls_top5"], epoch)
        self.writer.add_scalar("Predictions/mean", val_metrics["pred_mean"], epoch)
        self.writer.add_scalar("Predictions/std", val_metrics["pred_std"], epoch)
        self.writer.add_scalar("Predictions/min", val_metrics["pred_min"], epoch)
        self.writer.add_scalar("Predictions/max", val_metrics["pred_max"], epoch)
