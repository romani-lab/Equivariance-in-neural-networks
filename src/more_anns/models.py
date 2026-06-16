"""AlexNet model variants for classification and orientation tasks."""

import copy
import logging

import torch
from torch import nn
from torchvision.models import alexnet

log = logging.getLogger(__name__)

DEFAULT_NUM_CLASSES = 1000


class _TruncatedBackbone(nn.Module):
    """AlexNet backbone truncated at a given layer, preserving state_dict naming."""

    def __init__(self, features, avgpool=None, *, flatten=False, classifier=None):
        super().__init__()
        self.features = features
        self._flatten = flatten
        if avgpool is not None:
            self.avgpool = avgpool
        if classifier is not None:
            self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        if hasattr(self, "avgpool"):
            x = self.avgpool(x)
        if self._flatten:
            x = torch.flatten(x, 1)
        if hasattr(self, "classifier"):
            x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    """Unified AlexNet supporting classification and/or orientation tasks.

    AlexNet architecture::

        features.0:  Conv2d(3, 64, 11, stride=4, padding=2)
        features.1:  ReLU
        features.2:  MaxPool2d(3, stride=2)
        features.3:  Conv2d(64, 192, 5, padding=2)
        features.4:  ReLU
        features.5:  MaxPool2d(3, stride=2)
        features.6:  Conv2d(192, 384, 3, padding=1)
        features.7:  ReLU
        features.8:  Conv2d(384, 256, 3, padding=1)
        features.9:  ReLU
        features.10: Conv2d(256, 256, 3, padding=1)
        features.11: ReLU
        features.12: MaxPool2d(3, stride=2)
        avgpool:     AdaptiveAvgPool2d(6, 6)
        flattening (not part of modules)
        classifier.0: Dropout(0.5)
        classifier.1: Linear(9216, 4096)
        classifier.2: ReLU
        classifier.3: Dropout(0.5)
        classifier.4: Linear(4096, 4096)
        classifier.5: ReLU

    Supported tasks:
    - "classification": Single image classification (returns logits)
    - "orientation": Siamese orientation regression (returns orientation)
    - "absolute_orientation": Single image absolute rotation prediction (returns orientation)
    - "multitask": Both classification and orientation (returns logits1, logits2, orientation)

    For Siamese orientation, orientation_fusion controls how the two embeddings are combined:
    - "concat": Concatenate embeddings and regress through a head.
        With similarity_layer="input", concatenates raw images (6 channels) for early fusion.
    - "product": Element-wise product of L2-normalized embeddings, then regress through a head
    """

    EMBEDDING_DIM = 4096  # Fixed by AlexNet architecture

    def __init__(
        self,
        task: str = "classification",
        similarity_layer: str = "classifier.5",
        orientation_fusion: str = "concat",
        num_classes: int = 1000,
        orientation_output_dim: int = 2,
        orientation_activation: str = "linear",
        pretrained_weights: str | None = None,
        pretrained_layers: str | None = None,
    ):
        """Initialize AlexNet.

        Args:
            task: "classification", "orientation", "absolute_orientation", or "multitask"
            similarity_layer: Layer at which to split the backbone — layers after this
                become task-specific heads. Use "input" for early fusion (6-channel concat).
            orientation_fusion: How to combine Siamese embeddings — "concat" or "product"
            num_classes: Number of classification classes
            orientation_output_dim: Number of orientation output values (1 for cos, 2 for sin/cos)
            orientation_activation: Activation for orientation output ("linear" or "tanh")
            pretrained_weights: "imagenet" or path to a checkpoint file, None for random init
            pretrained_layers: Prefix filter for which layers to load (e.g. "features"), None for all
        """
        super().__init__()

        if task not in ("classification", "orientation", "absolute_orientation", "multitask"):
            msg = f"task must be 'classification', 'orientation', 'absolute_orientation', or 'multitask', got '{task}'"
            raise ValueError(msg)

        if orientation_fusion not in ("concat", "product"):
            msg = f"orientation_fusion must be 'concat' or 'product', got '{orientation_fusion}'"
            raise ValueError(msg)

        self.task = task
        self.orientation_fusion = orientation_fusion
        self.similarity_layer = similarity_layer

        # Derived flags for convenience
        self.num_images = 1 if task in ("classification", "absolute_orientation") else 2
        self.return_classification = task in ("classification", "multitask")
        self.return_orientation = task in ("orientation", "multitask", "absolute_orientation")

        # Build AlexNet and load pretrained weights before splitting
        base = alexnet(weights=None)
        self.backbone = base  # temporary — full AlexNet for pretrained loading
        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights, pretrained_layers)

        # Split backbone at similarity_layer
        original_features = list(base.features.children())
        original_classifier = list(base.classifier.children())

        if similarity_layer == "input":
            self.backbone = nn.Identity()
            remaining = [*original_features, nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(1), *original_classifier]
        elif similarity_layer.startswith("features."):
            idx = int(similarity_layer.split(".")[-1]) + 1
            self.backbone = _TruncatedBackbone(
                features=nn.Sequential(*original_features[:idx]),
            )
            remaining = [*original_features[idx:], nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(1), *original_classifier]
        elif similarity_layer == "avgpool":
            self.backbone = _TruncatedBackbone(
                features=base.features,
                avgpool=base.avgpool,
            )
            remaining = [nn.Flatten(1), *original_classifier]
        elif similarity_layer == "flatten":
            self.backbone = _TruncatedBackbone(
                features=base.features,
                avgpool=base.avgpool,
                flatten=True,
            )
            remaining = list(original_classifier)
        elif similarity_layer.startswith("classifier."):
            idx = int(similarity_layer.split(".")[-1]) + 1
            self.backbone = _TruncatedBackbone(
                features=base.features,
                avgpool=base.avgpool,
                flatten=True,
                classifier=nn.Sequential(*original_classifier[:idx]),
            )
            remaining = original_classifier[idx:]
        else:
            valid = (
                ["input"]
                + [f"features.{i}" for i in range(len(original_features))]
                + ["avgpool", "flatten"]
                + [f"classifier.{i}" for i in range(len(original_classifier))]
            )
            msg = f"similarity_layer '{similarity_layer}' not found. Valid: {valid}"
            raise ValueError(msg)

        # Split remaining into pre_combine (before first Linear) and post_combine (from first Linear)
        first_linear_idx = next(
            (i for i, layer in enumerate(remaining) if isinstance(layer, nn.Linear)),
            len(remaining),
        )
        pre_combine = remaining[:first_linear_idx]
        post_combine = remaining[first_linear_idx:]

        # Determine combination dimension (input to first Linear after pre_combine)
        if post_combine:
            combine_dim = post_combine[0].in_features
        else:
            with torch.no_grad():
                _d = self.backbone(torch.zeros(1, 3, 224, 224))
                combine_dim = _d.view(1, -1).shape[1]

        # Classification head: full remaining pipeline with adapted output
        if self.return_classification:
            cls_layers = list(remaining)
            if cls_layers:
                for i in range(len(cls_layers) - 1, -1, -1):
                    if isinstance(cls_layers[i], nn.Linear):
                        if num_classes != DEFAULT_NUM_CLASSES:
                            cls_layers[i] = nn.Linear(cls_layers[i].in_features, num_classes)
                        break
                self.classifier_head = nn.Sequential(*cls_layers) if len(cls_layers) > 1 else cls_layers[0]
            else:
                self.classifier_head = nn.Linear(combine_dim, num_classes)

        # Helper: mirror post_combine structure with adapted input/output dims
        def _mirror_post_combine(in_dim: int, out_dim: int) -> nn.Module:
            if not post_combine:
                layers = [nn.Linear(in_dim, out_dim)]
            else:
                layers: list[nn.Module] = []
                current_dim = in_dim
                for layer in post_combine[:-1]:
                    if isinstance(layer, nn.Linear):
                        layers.append(nn.Linear(current_dim, layer.out_features))
                        current_dim = layer.out_features
                    elif isinstance(layer, nn.ReLU):
                        layers.append(nn.ReLU(inplace=True))
                    elif isinstance(layer, nn.Dropout):
                        layers.append(nn.Dropout(p=layer.p))
                layers.append(nn.Linear(current_dim, out_dim))
            if orientation_activation == "tanh":
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        # Orientation encoder + head
        if self.return_orientation:
            if task == "absolute_orientation":
                # Full remaining pipeline (single image, no combination)
                abs_layers = [copy.deepcopy(layer) for layer in remaining]
                if abs_layers:
                    for i in range(len(abs_layers) - 1, -1, -1):
                        if isinstance(abs_layers[i], nn.Linear):
                            abs_layers[i] = nn.Linear(abs_layers[i].in_features, orientation_output_dim)
                            break
                else:
                    abs_layers = [nn.Linear(combine_dim, orientation_output_dim)]
                if orientation_activation == "tanh":
                    abs_layers.append(nn.Tanh())
                self.orientation_head = nn.Sequential(*abs_layers)
                self.ori_encoder = None
            elif orientation_fusion == "concat":
                if similarity_layer == "input":
                    # 6-channel early fusion: concat raw images, process through full network
                    self.ori_encoder = None
                    head_layers = [copy.deepcopy(layer) for layer in remaining]
                    old_conv = head_layers[0]
                    head_layers[0] = nn.Conv2d(
                        6,
                        old_conv.out_channels,
                        old_conv.kernel_size,
                        old_conv.stride,
                        old_conv.padding,
                    )
                    for i in range(len(head_layers) - 1, -1, -1):
                        if isinstance(head_layers[i], nn.Linear):
                            head_layers[i] = nn.Linear(head_layers[i].in_features, orientation_output_dim)
                            break
                    if orientation_activation == "tanh":
                        head_layers.append(nn.Tanh())
                    self.orientation_head = nn.Sequential(*head_layers)
                else:
                    self.ori_encoder = nn.Sequential(*copy.deepcopy(pre_combine)) if pre_combine else None
                    self.orientation_head = _mirror_post_combine(combine_dim * 2, orientation_output_dim)
            elif orientation_fusion == "product":
                self.ori_encoder = nn.Sequential(*copy.deepcopy(pre_combine)) if pre_combine else None
                self.orientation_head = _mirror_post_combine(combine_dim, orientation_output_dim)
        else:
            self.orientation_head = None
            self.ori_encoder = None

    def _load_pretrained(self, weights: str, layers: str | None = None) -> None:
        """Load pretrained weights into the backbone.

        Called before splitting, so self.backbone is a full AlexNet with standard
        state_dict keys (features.0.weight, classifier.1.weight, etc.).

        Args:
            weights: "imagenet" for torchvision weights, or path to a checkpoint file.
            layers: If set, only load keys starting with this prefix (e.g. "features").
        """
        if weights == "imagenet":
            src_state = alexnet(weights="DEFAULT").state_dict()
        else:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            src_state = ckpt.get("model_state_dict", ckpt)
            # Strip common prefixes from checkpoint keys
            src_state = {k.removeprefix("model.").removeprefix("backbone."): v for k, v in src_state.items()}

        if layers is not None:
            src_state = {k: v for k, v in src_state.items() if k.startswith(layers)}

        missing, unexpected = self.backbone.load_state_dict(src_state, strict=False)
        loaded = set(src_state.keys()) - set(unexpected)
        log.info(
            f"Loaded {len(loaded)} pretrained weight tensors, {len(missing)} missing, {len(unexpected)} unexpected"
        )

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            *args: Image tensors of shape (batch_size, 3, 224, 224).
                - classification/absolute_orientation: (x1,)
                - orientation/multitask: (x1, x2)

        Returns:
            Tuple of outputs based on task:
            - classification: (logits,)
            - orientation: (orientation,)
            - absolute_orientation: (orientation,)
            - multitask: (logits1, logits2, orientation)
        """
        if len(args) != self.num_images:
            msg = f"task='{self.task}' requires exactly {self.num_images} image(s), got {len(args)}"
            raise ValueError(msg)

        x1 = args[0]
        x2 = args[1] if len(args) > 1 else None

        # Forward pass through backbone
        emb1 = self.backbone(x1)
        emb2 = self.backbone(x2) if x2 is not None else None

        result = []

        # Classification
        if self.return_classification:
            result.append(self.classifier_head(emb1))
            if emb2 is not None:
                result.append(self.classifier_head(emb2))

        # Orientation
        if self.return_orientation:
            if self.task == "absolute_orientation":
                orientation = self.orientation_head(emb1)
            else:
                enc1 = self.ori_encoder(emb1) if self.ori_encoder is not None else emb1
                enc2 = self.ori_encoder(emb2) if self.ori_encoder is not None else emb2
                if self.orientation_fusion == "concat":
                    combined = torch.cat([enc1, enc2], dim=1)
                else:
                    # product: element-wise product of L2-normalized embeddings
                    enc1 = nn.functional.normalize(enc1, p=2, dim=1)
                    enc2 = nn.functional.normalize(enc2, p=2, dim=1)
                    combined = enc1 * enc2
                orientation = self.orientation_head(combined)
            result.append(orientation)

        return tuple(result)
