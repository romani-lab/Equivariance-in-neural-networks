"""Training script for Siamese AlexNet orientation regression (Task 2)."""

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from more_anns.data import ImageNetDataModule
from more_anns.losses import get_angle_representation, get_orientation_loss
from more_anns.metrics import orientation_mae
from more_anns.models import AlexNet
from more_anns.train_base import BaseTrainer

log = logging.getLogger(__name__)


class OrientationTrainer(BaseTrainer):
    """Trainer for AlexNet orientation regression task.

    Supports two model types:
    - "siamese": Process two images through shared backbone (use similarity_layer="input"
        with orientation_fusion="concat" for 6-channel early fusion)
    - "absolute": Process a single rotated image, predict its absolute rotation
    """

    def __init__(
        self,
        dataset: str,
        output_dir: str | Path,
        *,
        use_circular_mask: bool = True,
        model_type: str = "siamese",
        orientation_fusion: str = "concat",
        similarity_layer: str = "classifier.5",
        angle_representation: str = "sincos_unordered",
        orientation_activation: str = "linear",
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

        self.model_type = model_type

        # Data
        use_pair = model_type != "absolute"
        self.data_module = ImageNetDataModule(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            return_pair=use_pair,
            return_label=False,
            return_angles=True,
            use_rotation_aug=not use_pair,
            use_circular_mask=use_circular_mask,
            rotation_quantization=rotation_quantization,
            interpolation_order=interpolation_order,
        )

        # Loss and model
        representation = get_angle_representation(angle_representation)
        self.criterion = get_orientation_loss(angle_representation, loss_type)

        if model_type == "siamese":
            self.model = AlexNet(
                task="orientation",
                orientation_fusion=orientation_fusion,
                similarity_layer=similarity_layer,
                orientation_output_dim=representation.output_dim,
                orientation_activation=orientation_activation,
                pretrained_weights=pretrained_weights,
                pretrained_layers=pretrained_layers,
            )
        elif model_type == "absolute":
            self.model = AlexNet(
                task="absolute_orientation",
                orientation_output_dim=representation.output_dim,
                orientation_activation=orientation_activation,
                pretrained_weights=pretrained_weights,
                pretrained_layers=pretrained_layers,
            )
        else:
            msg = f"model_type must be 'siamese' or 'absolute', got '{model_type}'"
            raise ValueError(msg)

        self.config = {
            "dataset": dataset,
            "use_circular_mask": use_circular_mask,
            "model_type": model_type,
            "orientation_fusion": orientation_fusion,
            "similarity_layer": similarity_layer,
            "angle_representation": angle_representation,
            "orientation_activation": orientation_activation,
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
        rep = self.criterion.representation

        metrics = {"loss": 0.0, "mae_unordered": 0.0}
        if rep.ordered:
            metrics["mae_ordered"] = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        for batch in pbar:
            if self.model_type == "absolute":
                img, angle = batch
                img = img.to(self.device)
                angle1 = torch.zeros_like(angle)
                angle2 = angle.to(self.device, dtype=torch.float32)
                angle1 = angle1.to(self.device, dtype=torch.float32)
                (sincos_pred,) = self.model(img)
            else:
                img1, img2, angle1, angle2 = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                angle1 = angle1.to(self.device, dtype=torch.float32)
                angle2 = angle2.to(self.device, dtype=torch.float32)
                (sincos_pred,) = self.model(img1, img2)

            loss = self.criterion(sincos_pred, angle1, angle2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = sincos_pred.size(0)
            metrics["loss"] += loss.item() * batch_size
            with torch.no_grad():
                mae_unord = orientation_mae(sincos_pred, angle1, angle2, rep, ordered=False).item()
                metrics["mae_unordered"] += mae_unord * batch_size
                if rep.ordered:
                    mae_ord = orientation_mae(sincos_pred, angle1, angle2, rep, ordered=True).item()
                    metrics["mae_ordered"] += mae_ord * batch_size
            total += batch_size

            pbar.set_postfix({"loss": loss.item(), "mae": metrics["mae_unordered"] / total})

        for k in metrics:
            metrics[k] /= total
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        val_loader = self.data_module.val_dataloader()
        rep = self.criterion.representation

        metrics = {"loss": 0.0, "mae_unordered": 0.0}
        if rep.ordered:
            metrics["mae_ordered"] = 0.0
        total = 0
        all_pred_angles = []
        all_target_angles = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")
        for batch in pbar:
            if self.model_type == "absolute":
                img, angle = batch
                img = img.to(self.device)
                angle1 = torch.zeros_like(angle).to(self.device, dtype=torch.float32)
                angle2 = angle.to(self.device, dtype=torch.float32)
                (sincos_pred,) = self.model(img)
            else:
                img1, img2, angle1, angle2 = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                angle1 = angle1.to(self.device, dtype=torch.float32)
                angle2 = angle2.to(self.device, dtype=torch.float32)
                (sincos_pred,) = self.model(img1, img2)

            loss = self.criterion(sincos_pred, angle1, angle2)

            # Collect decoded angles for distribution analysis
            pred_deg = rep.decode(sincos_pred)
            if rep.ordered:
                target_deg = (angle2 - angle1) % 360
            else:
                delta = torch.abs(angle2 - angle1)
                target_deg = torch.min(delta, 360 - delta)
            all_pred_angles.append(pred_deg.cpu())
            all_target_angles.append(target_deg.cpu())

            batch_size = sincos_pred.size(0)
            metrics["loss"] += loss.item() * batch_size
            mae_unord = orientation_mae(sincos_pred, angle1, angle2, rep, ordered=False).item()
            metrics["mae_unordered"] += mae_unord * batch_size
            if rep.ordered:
                mae_ord = orientation_mae(sincos_pred, angle1, angle2, rep, ordered=True).item()
                metrics["mae_ordered"] += mae_ord * batch_size
            total += batch_size

            pbar.set_postfix({"loss": loss.item(), "mae": metrics["mae_unordered"] / total})

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
        log.info(
            f"  Predictions - mean: {metrics['pred_mean']:.1f}°, std: {metrics['pred_std']:.1f}°, "
            f"min: {metrics['pred_min']:.1f}°, max: {metrics['pred_max']:.1f}°"
        )

        return metrics

    def _is_best(self, val_metrics, best):
        mae = val_metrics["mae_unordered"]
        is_better = best is None or mae < best
        new_best = mae if is_better else best
        return is_better, new_best, f"MAE: {new_best:.2f}°"

    def _log_metrics(self, train_metrics, val_metrics, epoch):
        rep = self.criterion.representation

        msg = f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae_unordered']:.2f}°"
        if rep.ordered:
            msg += f", MAE (ordered): {train_metrics['mae_ordered']:.2f}°"
        log.info(msg)

        msg = f"Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae_unordered']:.2f}°"
        if rep.ordered:
            msg += f", MAE (ordered): {val_metrics['mae_ordered']:.2f}°"
        log.info(msg)

        self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        self.writer.add_scalar("MAE/train", train_metrics["mae_unordered"], epoch)
        self.writer.add_scalar("MAE/val", val_metrics["mae_unordered"], epoch)
        if rep.ordered:
            self.writer.add_scalar("MAE_ordered/train", train_metrics["mae_ordered"], epoch)
            self.writer.add_scalar("MAE_ordered/val", val_metrics["mae_ordered"], epoch)
        self.writer.add_scalar("Predictions/mean", val_metrics["pred_mean"], epoch)
        self.writer.add_scalar("Predictions/std", val_metrics["pred_std"], epoch)
        self.writer.add_scalar("Predictions/min", val_metrics["pred_min"], epoch)
        self.writer.add_scalar("Predictions/max", val_metrics["pred_max"], epoch)
