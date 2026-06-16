"""Feature extraction API for trained models.

Usage::

    from more_anns.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor("path/to/best_model.pth")
    print(extractor.layer_names)

    # Single image (classification or absolute orientation)
    activations = extractor("image.jpg", layers=["features.6", "classifier.4"])

    # Image pair (siamese orientation or multitask)
    activations = extractor(img1, img2, layers=["backbone.features.6"])

    # All layers
    activations = extractor("image.jpg")
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from more_anns.losses import get_angle_representation
from more_anns.models import AlexNet
from more_anns.preprocessing import CircularMaskTransform


class FeatureExtractor:
    """Load a trained model and extract intermediate activations."""

    def __init__(self, checkpoint_path: str | Path):
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to a .pth checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.config = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self._build_model(cfg)
        self._load_weights(self.model, ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._transform = self._build_transform(cfg)

    @staticmethod
    def _build_model(cfg: dict) -> AlexNet:
        """Reconstruct model from checkpoint config."""
        is_classification = "angle_representation" not in cfg

        if is_classification:
            return AlexNet(task="classification")

        rep = get_angle_representation(cfg["angle_representation"])

        model_type = cfg.get("model_type", "siamese")
        is_multitask = "classification_weight" in cfg or "orientation_weight" in cfg

        if model_type == "siamese":
            task = "multitask" if is_multitask else "orientation"
            return AlexNet(
                task=task,
                orientation_fusion=cfg.get("orientation_fusion", "concat"),
                similarity_layer=cfg.get("similarity_layer", "classifier.5"),
                orientation_output_dim=rep.output_dim,
                orientation_activation=cfg.get("orientation_activation", "linear"),
            )
        elif model_type == "absolute":
            return AlexNet(
                task="absolute_orientation",
                orientation_output_dim=rep.output_dim,
                orientation_activation=cfg.get("orientation_activation", "linear"),
            )
        else:
            msg = f"Unknown model_type: {model_type}"
            raise ValueError(msg)

    @staticmethod
    def _load_weights(model: AlexNet, ckpt: dict) -> None:
        """Load checkpoint weights."""
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict)


    @staticmethod
    def _build_transform(cfg: dict) -> transforms.Compose:
        """Build preprocessing pipeline from config."""
        tfms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        if cfg.get("use_circular_mask", True):
            tfms.append(CircularMaskTransform(size=224, fill_value=128))
        tfms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(tfms)

    @property
    def layer_names(self) -> list[str]:
        """All hookable layer names in the model."""
        return [name for name, _ in self.model.named_modules() if name]

    def _to_tensor(self, image) -> torch.Tensor:
        """Convert input to a preprocessed tensor with batch dim."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self._transform(image)
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # noqa: PLR2004
                image = image.unsqueeze(0)
            return image.to(self.device)
        msg = f"Expected str, Path, PIL.Image, or Tensor, got {type(image)}"
        raise TypeError(msg)

    def __call__(
        self,
        img1,
        img2=None,
        layers: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract activations from the model.

        Args:
            img1: Image as file path (str/Path), PIL Image, or tensor.
            img2: Optional second image for siamese/multitask models.
                If omitted for a siamese/multitask model, only the shared
                backbone is executed (orientation/classifier heads are skipped).
            layers: Layer names to capture. If None, captures all layers.

        Returns:
            Dict mapping layer name to activation tensor.
        """
        x1 = self._to_tensor(img1)
        x2 = self._to_tensor(img2) if img2 is not None else None

        target_layers = set(layers) if layers is not None else None
        activations = {}
        hooks = []

        for name, module in self.model.named_modules():
            if not name:
                continue
            if target_layers is not None and name not in target_layers:
                continue

            def _hook(_mod, _inp, out, name=name):
                if isinstance(out, torch.Tensor):
                    activations[name] = out.detach()

            hooks.append(module.register_forward_hook(_hook))

        with torch.no_grad():
            if x2 is not None:
                self.model(x1, x2)
            elif self.model.num_images == 1:
                self.model(x1)
            else:
                # Siamese/multitask model with single image: run backbone only
                self.model.backbone(x1)

        for h in hooks:
            h.remove()

        return activations

    def extract_activations(
        self,
        img_all: np.ndarray,
        layers: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract activations for a batch of pre-transformed images.

        This method mirrors the interface of ``alexnet_responses`` in
        ``model_loader.py``.  Images must already be transformed (e.g. via
        ``standard_img_transform``) — no resizing, cropping, or normalisation
        is applied internally.

        Args:
            img_all: NumPy array of shape ``(n_images, 3, H, W)`` containing
                pre-transformed images (float32, already normalised).
            layers: Layer names to capture.  If *None*, captures all layers.

        Returns:
            Dict mapping each layer name to a NumPy array of shape
            ``(n_images, C, *spatial)`` containing the activations for every
            image.
        """
        target_layers = set(layers) if layers is not None else None
        activations: dict[str, list[torch.Tensor]] = {}

        # Register hooks once
        hooks = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            if target_layers is not None and name not in target_layers:
                continue

            def _hook(_mod, _inp, out, name=name):
                if isinstance(out, torch.Tensor):
                    activations.setdefault(name, []).append(out.detach().cpu())

            hooks.append(module.register_forward_hook(_hook))

        # Forward pass for each image (no transforms applied)
        with torch.no_grad():
            for i in range(img_all.shape[0]):
                x = torch.from_numpy(img_all[i]).unsqueeze(0).to(self.device)
                if self.model.num_images == 1:
                    self.model(x)
                else:
                    self.model.backbone(x)

        for h in hooks:
            h.remove()

        # Stack per-image activations into a single array per layer
        return {name: torch.cat(tensors, dim=0).numpy() for name, tensors in activations.items()}
