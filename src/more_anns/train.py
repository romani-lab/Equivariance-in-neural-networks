"""Hydra-based training script for AlexNet variants."""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from more_anns.train_classification import ClassificationTrainer
from more_anns.train_multitask import MultiTaskTrainer
from more_anns.train_orientation import OrientationTrainer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training entry point using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    log.info("Starting training with configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Get output directory from Hydra
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Save full config to output directory
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Saved configuration to {config_path}")

    # Dispatch to appropriate trainer based on task
    task_name = cfg.task.name

    if task_name == "classification":
        trainer = ClassificationTrainer(
            dataset=cfg.data.dataset,
            output_dir=output_dir,
            use_circular_mask=cfg.task.use_circular_mask,
            use_rotation_aug=cfg.task.use_rotation_aug,
            rotation_quantization=cfg.task.rotation_quantization,
            interpolation_order=cfg.task.interpolation_order,
            num_epochs=cfg.training.num_epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
            num_workers=cfg.data.num_workers,
            device=cfg.training.device,
            pretrained_weights=cfg.task.pretrained_weights,
            pretrained_layers=cfg.task.pretrained_layers,
            freeze_backbone=cfg.task.freeze_backbone,
            unfreeze_at_epoch=cfg.task.unfreeze_at_epoch,
            hf_repo_id=cfg.hf_repo_id,
            backbone=cfg.task.get("backbone", "alexnet"),
        )
        trainer.train()

    elif task_name == "orientation":
        trainer = OrientationTrainer(
            dataset=cfg.data.dataset,
            output_dir=output_dir,
            use_circular_mask=cfg.task.use_circular_mask,
            model_type=cfg.task.model_type,
            orientation_fusion=cfg.task.orientation_fusion,
            similarity_layer=cfg.task.similarity_layer,
            angle_representation=cfg.task.angle_representation,
            orientation_activation=cfg.task.orientation_activation,
            loss_type=cfg.task.loss_type,
            rotation_quantization=cfg.task.rotation_quantization,
            interpolation_order=cfg.task.interpolation_order,
            num_epochs=cfg.training.num_epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
            num_workers=cfg.data.num_workers,
            device=cfg.training.device,
            pretrained_weights=cfg.task.pretrained_weights,
            pretrained_layers=cfg.task.pretrained_layers,
            freeze_backbone=cfg.task.freeze_backbone,
            unfreeze_at_epoch=cfg.task.unfreeze_at_epoch,
            hf_repo_id=cfg.hf_repo_id,
        )
        trainer.train()

    elif task_name == "multitask":
        trainer = MultiTaskTrainer(
            dataset=cfg.data.dataset,
            output_dir=output_dir,
            use_circular_mask=cfg.task.use_circular_mask,
            classification_weight=cfg.task.classification_weight,
            orientation_weight=cfg.task.orientation_weight,
            angle_representation=cfg.task.angle_representation,
            orientation_activation=cfg.task.orientation_activation,
            orientation_fusion=cfg.task.orientation_fusion,
            similarity_layer=cfg.task.similarity_layer,
            loss_type=cfg.task.loss_type,
            rotation_quantization=cfg.task.rotation_quantization,
            interpolation_order=cfg.task.interpolation_order,
            num_epochs=cfg.training.num_epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
            num_workers=cfg.data.num_workers,
            device=cfg.training.device,
            pretrained_weights=cfg.task.pretrained_weights,
            pretrained_layers=cfg.task.pretrained_layers,
            freeze_backbone=cfg.task.freeze_backbone,
            unfreeze_at_epoch=cfg.task.unfreeze_at_epoch,
            hf_repo_id=cfg.hf_repo_id,
        )
        trainer.train()

    else:
        msg = f"Unknown task: {task_name}"
        raise ValueError(msg)

    log.info(f"Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    train()
