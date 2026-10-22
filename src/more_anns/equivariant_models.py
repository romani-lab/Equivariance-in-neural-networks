"""AlexNet-style CNNs with built-in rotation equivariance (escnn, regular fields of C_15),
the three networks of Supplementary Fig. 6.

Variants
--------
- ``'C15'``       : exact C_15-steerable network. Every conv layer carries regular fields of
                    C_15, so rotations of the (circularly masked) input by multiples of
                    24 deg = 360/15 about the image centre are exact symmetries of every layer
                    (up to interpolation / sampling).
- ``'C15>C5>C1'`` : symmetry relaxed across depth. The group is restricted along the
                    hierarchy: conv1-2 over C_15 (exact to 24 deg), conv3-4 over C_5 (exact to
                    multiples of 72 deg), conv5 over the trivial group (no built-in symmetry).
                    Because 15 = 3 * 5 is square-free, C_5 and C_3 cannot both appear in one
                    nested chain, so C15 > C5 > C1 is the realisable graded schedule.
- ``'C1'``        : trivial group at every layer, i.e. NO built-in rotation equivariance -- an
                    ordinary CNN that keeps escnn's band-limited steerable filter basis, the
                    anti-aliased pooling and every other detail of the two networks above, so
                    it isolates the effect of the symmetry itself. Channels equal AlexNet's
                    exactly (64/192/384/256/256) and GroupPooling is a no-op, so the head
                    matches AlexNet's.

All three are trained on ImageNet-1k classification with the recipe of the default AlexNet
(docs/ann_training.md) and analysed with the same pipeline as the other networks; the
population responses are taken from the convolutional layers before group pooling.

Layer / index convention (matches torchvision AlexNet, so utils.model_loader's ``layer_dict``
needs no change: conv1..conv5 = ``backbone.features.{0,3,6,8,10}``)::

    features.0   R2Conv  k11 p5 s1   conv1   (analysed)  on the 56-px input_pool output
    features.1   ReLU
    features.2   AvgPoolAntialiased s2
    features.3   R2Conv  k5  p2 s1   conv2   (analysed)
    features.4   ReLU
    features.5   AvgPoolAntialiased s2 [+ restrict C15->C5 for the relaxed variant]
    features.6   R2Conv  k3  p1 s1   conv3   (analysed)
    features.7   ReLU
    features.8   R2Conv  k3  p1 s1   conv4   (analysed)
    features.9   ReLU                [+ restrict C5->C1 for the relaxed variant]
    features.10  R2Conv  k3  p1 s1   conv5   (analysed)
    features.11  ReLU
    features.12  AvgPoolAntialiased s2

To retain equivariance across layers there are no strided convolutions and no max-pooling:
downsampling between convolutions is anti-aliased average pooling (stride 2), and the
224 x 224 input is first anti-aliased-downsampled to 56 x 56 (``input_pool``, two symmetric
stride-2 anti-aliased pools) so that conv1 can run at stride 1. The anti-aliasing also removes
the high-frequency image content that dominates the residual (interpolation) equivariance
error of a sampled CNN on discretely rotated images. Analysed map sizes: 56/28/14/14/14 px,
comparable to AlexNet's 55/27/13/13/13.

Channel counts are matched to AlexNet per layer (round(alexnet_channels / group_order)):

======  ==========  ==================  ==================  ==================
layer   AlexNet ch  C15                 C15>C5>C1           C1
======  ==========  ==================  ==================  ==================
conv1   64          4 x C15 = 60        4 x C15 = 60        64 x C1 = 64
conv2   192         13 x C15 = 195      13 x C15 = 195      192
conv3   384         26 x C15 = 390      77 x C5  = 385      384
conv4   256         17 x C15 = 255      51 x C5  = 255      256
conv5   256         17 x C15 = 255      255 x C1 = 255      256
======  ==========  ==================  ==================  ==================

For classification, the final feature maps are collapsed to one rotation-invariant channel
per field by group pooling, followed by adaptive average pooling and a fully connected head
of the same design as AlexNet (two 4096-unit hidden layers).

The hook constraint: the analysis pipeline (more_anns.feature_extractor.FeatureExtractor)
captures activations with a forward hook that only records ``torch.Tensor`` outputs, but
escnn modules emit ``GeometricTensor``. Every block is therefore wrapped in ``_EqBlock``,
which takes a plain tensor, re-wraps it as a GeometricTensor of the (known) input field
type, runs the escnn op, and returns ``.tensor`` -- so the hook at each conv index captures
a plain ``(1, C, H, W)`` tensor while the field-type metadata is threaded through explicitly.
"""

from __future__ import annotations

import torch
from torch import nn

from escnn import gspaces, nn as enn

# torchvision AlexNet per-layer output channels (conv1..conv5)
ALEXNET_CHANNELS = (64, 192, 384, 256, 256)
KERNELS = (11, 5, 3, 3, 3)
PADDINGS = (5, 2, 1, 1, 1)
INPUT_SIZE = 224          # size of the pipeline stimulus tensor
INTERNAL_SIZE = 56        # conv1 resolution; input_pool downsamples 224 -> 56
POOL_SIGMA = 0.66

# Per-variant group order at each conv layer (conv1..conv5)
VARIANT_ORDERS = {
    "C15": (15, 15, 15, 15, 15),
    "C15>C5>C1": (15, 15, 5, 5, 1),
    "C1": (1, 1, 1, 1, 1),
}

# checkpoint backbone key -> variant (the 'backbone' entry of a checkpoint's config)
BACKBONES = {
    "eqv_c15": "C15",
    "eqv_c15_restricted": "C15>C5>C1",
    "eqv_c1": "C1",
}


class _EqBlock(nn.Module):
    """Wrap an escnn EquivariantModule so it maps a plain tensor to a plain tensor.

    ``in_type`` is the field type of the incoming activation; the block re-wraps the
    tensor, applies the escnn op, and unwraps the result. This keeps every block boundary
    a plain ``torch.Tensor`` so ``FeatureExtractor``'s forward hooks capture the conv
    outputs, while the geometric structure is preserved by threading the field types.
    """

    def __init__(self, op: enn.EquivariantModule, in_type: enn.FieldType):
        super().__init__()
        self.op = op
        self.in_type = in_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(enn.GeometricTensor(x, self.in_type)).tensor


def _regular_type(gspace, n_fields):
    """Field type of ``n_fields`` regular fields (trivial fields when the group is C_1)."""
    if gspace.fibergroup.order() == 1:
        return enn.FieldType(gspace, n_fields * [gspace.trivial_repr])
    return enn.FieldType(gspace, n_fields * [gspace.regular_repr])


def _n_fields(alexnet_channels, order):
    """Number of regular fields so that fields*order ~= alexnet_channels (>=1)."""
    return max(1, int(round(alexnet_channels / order)))


class EquivariantAlexNet(nn.Module):
    """AlexNet-shaped CNN with built-in C_15 rotation equivariance (escnn backbone).

    Parameters
    ----------
    variant : str
        ``'C15'`` (exact), ``'C15>C5>C1'`` (relaxed across depth) or ``'C1'`` (none).
    num_classes : int
        Size of the classification head.
    n_field_group : int
        Group order N of the top group (15). Kept as an argument for record only; the
        analysis needs N = 15 (the 24 deg stimulus step).

    Attributes exposed for the rest of the code base
    ------------------------------------------------
    - ``num_images = 1``          : FeatureExtractor dispatches single-image models with this.
    - ``backbone.features``       : nn.Sequential; indices 0/3/6/8/10 are conv1..conv5.
    - ``field_types``             : dict layer_name -> escnn FieldType of that conv output.
    """

    def __init__(self, variant: str = "C15", num_classes: int = 1000, n_field_group: int = 15):
        super().__init__()
        if variant not in VARIANT_ORDERS:
            raise ValueError(f"variant must be one of {list(VARIANT_ORDERS)}, got {variant!r}")
        if n_field_group != 15:
            raise ValueError("n_field_group must be 15 (matches the 24 deg = 360/15 stimulus step)")

        self.variant = variant
        self.num_classes = num_classes
        self.num_images = 1  # FeatureExtractor single-image dispatch
        self.internal_size = INTERNAL_SIZE

        orders = VARIANT_ORDERS[variant]
        self.gspace = gspaces.rot2dOnR2(N=n_field_group)

        # --- gspaces per conv layer -------------------------------------------------
        # The symmetry group can be restricted along depth (C15>C5>C1) and, for the
        # all-trivial 'C1' variant, already at conv1. The ambient group is always C_15
        # (self.gspace) so C_5 / C_1 enter as its subgroups and the 24 deg = 360/15
        # analysis step stays defined.
        gspaces_per_layer = []
        gsp = self.gspace
        for i in range(5):
            prev = orders[i - 1] if i > 0 else n_field_group
            if orders[i] != prev:
                gsp, _, _ = gsp.restrict(orders[i])
            gspaces_per_layer.append(gsp)

        # conv1 input (3 colour channels = trivial reps) lives on conv1's own gspace, so the
        # first R2Conv has matching in/out groups (== self.gspace unless conv1 is restricted).
        first_gspace = gspaces_per_layer[0]
        in_type = enn.FieldType(first_gspace, 3 * [first_gspace.trivial_repr])
        self.input_type = in_type
        # --- input downsampling 224 -> 56 (two symmetric anti-aliased stride-2 pools) ---
        self.input_pool = nn.Sequential(
            _EqBlock(enn.PointwiseAvgPoolAntialiased(in_type, sigma=POOL_SIGMA, stride=2), in_type),
            _EqBlock(enn.PointwiseAvgPoolAntialiased(in_type, sigma=POOL_SIGMA, stride=2), in_type),
        )

        # --- conv stack --------------------------------------------------------------
        n_fields = [_n_fields(ALEXNET_CHANNELS[i], orders[i]) for i in range(5)]
        conv_out_types = [_regular_type(gspaces_per_layer[i], n_fields[i]) for i in range(5)]

        blocks: list[nn.Module] = []
        self.field_types: dict[str, enn.FieldType] = {}
        cur_type = in_type  # type flowing into the next block (plain-tensor channels tagged)

        def add_conv(i, cur_type):
            out_type = conv_out_types[i]
            conv = enn.R2Conv(cur_type, out_type, kernel_size=KERNELS[i], padding=PADDINGS[i], stride=1)
            blocks.append(_EqBlock(conv, cur_type))
            self.field_types[f"conv{i + 1}"] = out_type
            return out_type

        def add_relu(t, restrict_to=None):
            relu = enn.ReLU(t)
            if restrict_to is None:
                blocks.append(_EqBlock(relu, t))
                return t
            r = enn.RestrictionModule(t, restrict_to)   # folded into the same block index
            blocks.append(_EqBlock(enn.SequentialModule(relu, r), t))
            return r.out_type

        def add_pool(t, restrict_to=None):
            pool = enn.PointwiseAvgPoolAntialiased(t, sigma=POOL_SIGMA, stride=2)
            if restrict_to is None:
                blocks.append(_EqBlock(pool, t))
                return t
            r = enn.RestrictionModule(t, restrict_to)   # folded into the same block index
            blocks.append(_EqBlock(enn.SequentialModule(pool, r), t))
            return r.out_type

        # Group restriction is folded into the pool/relu block that PRECEDES the next conv,
        # so conv1..conv5 keep the exact indices 0/3/6/8/10 (no extra block is inserted).
        restrict_before_conv3 = orders[2] if orders[2] != orders[1] else None
        restrict_before_conv5 = orders[4] if orders[4] != orders[3] else None

        # conv1 (0), relu (1), pool (2)
        cur_type = add_conv(0, cur_type); add_relu(cur_type); add_pool(cur_type)
        # conv2 (3), relu (4), pool [+restrict->conv3 group] (5)
        cur_type = add_conv(1, cur_type); add_relu(cur_type)
        cur_type = add_pool(cur_type, restrict_to=restrict_before_conv3)
        # conv3 (6), relu (7)
        cur_type = add_conv(2, cur_type); add_relu(cur_type)
        # conv4 (8), relu [+restrict->conv5 group] (9)
        cur_type = add_conv(3, cur_type)
        cur_type = add_relu(cur_type, restrict_to=restrict_before_conv5)
        # conv5 (10), relu (11), pool (12)
        cur_type = add_conv(4, cur_type); add_relu(cur_type); add_pool(cur_type)

        self.backbone = _EqBackbone(nn.Sequential(*blocks))
        self._head_in_type = cur_type  # field type after features (== conv5 out type)

        # --- classification head: group pooling -> invariant FC head -------------------
        # GroupPooling collapses each field to one rotation-invariant channel.
        self.group_pool = enn.GroupPooling(self._head_in_type)
        n_inv = self.group_pool.out_type.size  # == n_fields[4]
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_inv * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def features_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run input downsampling + the conv stack; returns the plain tensor after features.12."""
        h = self.input_pool(x)
        return self.backbone.features(h)

    def forward(self, x: torch.Tensor):
        """Classification forward. Returns ``(logits,)`` (tuple) to match ClassificationTrainer."""
        h = self.features_forward(x)
        g = enn.GeometricTensor(h, self._head_in_type)
        inv = self.group_pool(g).tensor
        inv = self.avgpool(inv)
        inv = torch.flatten(inv, 1)
        return (self.classifier(inv),)

    def config(self) -> dict:
        """Checkpoint config that FeatureExtractor._build_model can rebuild from."""
        backbone = next(k for k, v in BACKBONES.items() if v == self.variant)
        return {
            "backbone": backbone,
            "variant": self.variant,
            "num_classes": self.num_classes,
            "group_order": 15,
            "input_size": INPUT_SIZE,
            "internal_size": self.internal_size,
            "use_circular_mask": True,
        }


class _EqBackbone(nn.Module):
    """Holds ``features`` so module names are ``backbone.features.<idx>`` (AlexNet-compatible)."""

    def __init__(self, features: nn.Sequential):
        super().__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)


def is_equivariant_backbone(backbone: str) -> bool:
    """True for the checkpoint ``backbone`` keys handled by :func:`build_equivariant_alexnet`."""
    return backbone in BACKBONES


def build_equivariant_alexnet(backbone: str, num_classes: int = 1000) -> EquivariantAlexNet:
    """Build from the checkpoint ``backbone`` key (one of :data:`BACKBONES`)."""
    if backbone not in BACKBONES:
        raise ValueError(f"unknown equivariant backbone {backbone!r} (expected one of {list(BACKBONES)})")
    return EquivariantAlexNet(variant=BACKBONES[backbone], num_classes=num_classes)


if __name__ == "__main__":
    # Smoke test: build every backbone, run a forward pass, report analysed-layer shapes.
    for backbone, variant in BACKBONES.items():
        m = EquivariantAlexNet(variant=variant)
        assert m.config()["backbone"] == backbone, (m.config()["backbone"], backbone)
        m.eval()
        captured = {}
        layer_dict = {"conv1": "features.0", "conv2": "features.3", "conv3": "features.6",
                      "conv4": "features.8", "conv5": "features.10"}
        hooks = []
        for lname, mod_name in layer_dict.items():
            idx = int(mod_name.split(".")[1])
            hooks.append(m.backbone.features[idx].register_forward_hook(
                lambda mod, i, o, n=lname: captured.__setitem__(n, tuple(o.shape))))
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            (logits,) = m(x)
        for h in hooks:
            h.remove()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"[{backbone}: {variant}] logits {tuple(logits.shape)}, {n_params / 1e6:.2f} M parameters")
        for lname in layer_dict:
            print(f"   {lname}: {captured[lname]}")
