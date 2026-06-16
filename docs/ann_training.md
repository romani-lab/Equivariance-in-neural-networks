# Training AlexNet variants for rotation equivariance analysis

This document describes how to train the three AlexNet variants used in the paper. All models are trained on ImageNet-1k for 90 epochs using SGD with momentum.

## Overview

| Model | Task | Description | Key result |
|-------|------|-------------|------------|
| **rot-aug** | classification | Standard AlexNet with rotation augmentation during training | 46.72% Top-1 (vs 54.08% without augmentation) |
| **ori** | orientation | Siamese AlexNet that predicts relative rotation between two images | 0.99° ordered MAE |
| **multi-task** | classification + orientation | Joint training on both tasks simultaneously | 49.52% Top-1, 5.32° ordered MAE |

All three use:
- Circular mask on input images
- Bilinear interpolation for rotations
- Ordered angle representation (`sincos_ordered`): predicts `(sin(angle), cos(angle))` for the full [0°, 360°) range
- MSE loss on the angle representation

## Requirements

- Python environment with dependencies from `pyproject.toml`
- ImageNet-1k dataset (via HuggingFace `ILSVRC/imagenet-1k`)
- GPU with sufficient memory (A100 recommended; training takes ~24–38h per model)

## Training commands

Training uses [Hydra](https://hydra.cc/) for configuration. Run from the project root:

```bash
python -m more_anns.train task=<task_config> [overrides...]
```

### rot-aug (classification with rotation augmentation)

```bash
python -m more_anns.train \
  task=classification \
  experiment=rot-aug \
  task.use_rotation_aug=true \
  task.use_circular_mask=true \
  training.num_epochs=90 \
  training.batch_size=256 \
  optimizer.lr=0.01 \
  training.device=cuda
```

This trains a standard AlexNet classifier on ImageNet with random rotation applied to training images. The rotation augmentation forces the backbone toward rotation-invariant representations.

**Architecture:** Single-image AlexNet with 1000-class output head.

### ori (orientation-only, ordered)

```bash
python -m more_anns.train \
  task=orientation \
  experiment=ori \
  task.model_type=siamese \
  task.orientation_fusion=concat \
  task.similarity_layer=classifier.3 \
  task.use_circular_mask=true \
  task.loss_type=mse \
  training.num_epochs=90 \
  training.batch_size=256 \
  optimizer.lr=0.01 \
  training.device=cuda
```

This trains a Siamese AlexNet to predict the relative rotation angle between two images of the same object. The two images share a backbone; their embeddings are concatenated and passed through an MLP head (split at `classifier.3` means layers 3+ become the orientation head: `Linear(8192, 4096) → ReLU → Linear(4096, 2)`).

**Architecture:** Shared AlexNet backbone up to `classifier.3`, then MLP orientation head outputting 2 values (sin, cos).

### multi-task (classification + ordered orientation)

```bash
python -m more_anns.train \
  task=multitask \
  experiment=multi-task \
  task.orientation_fusion=concat \
  task.similarity_layer=classifier.3 \
  task.classification_weight=1.0 \
  task.orientation_weight=1.0 \
  task.use_circular_mask=true \
  task.loss_type=mse \
  training.num_epochs=90 \
  training.batch_size=256 \
  optimizer.lr=0.01 \
  training.device=cuda
```

This trains a single backbone on both tasks simultaneously. The total loss is:

```
L = classification_weight * L_cls + orientation_weight * L_ori
```

The backbone is shared; `classifier.3` onward splits into a classification head (1000 classes) and an orientation head (2 outputs for sin/cos). At each batch, two randomly-rotated views of each image are generated: the classification head processes each view independently, while the orientation head receives the concatenated pair.

**Architecture:** Shared AlexNet backbone up to `classifier.3`, then parallel classification head (1000 classes) and MLP orientation head (2 outputs).

## Configuration reference

Key parameters and their defaults (from `src/more_anns/configs/`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task.use_circular_mask` | `true` | Apply circular mask to remove corner artifacts from rotation |
| `task.use_rotation_aug` | `false` | Random rotation augmentation (classification only) |
| `task.similarity_layer` | `classifier.5` | Where to split backbone from task head (deeper = more shared) |
| `task.orientation_fusion` | `concat` | How to combine Siamese embeddings (`concat` or `product`) |
| `task.angle_representation` | `sincos_ordered` | `sincos_ordered` for [0°,360°), `sincos_unordered` for [0°,180°] |
| `task.loss_type` | `mse` | Loss on angle representation (`mse` or `cosine`) |
| `task.classification_weight` | `1.0` | Weight for classification loss (multitask only) |
| `task.orientation_weight` | `1.0` | Weight for orientation loss (multitask only) |
| `task.rotation_quantization` | `null` | If set (e.g., 24), quantize angles to N bins |
| `training.batch_size` | `256` | Batch size |
| `optimizer.lr` | `0.01` | Learning rate (scale linearly with batch size if >256) |

## Output

Each training run saves to `cache/more_anns/<experiment>/`:
- `config.yaml` — full resolved configuration
- `best_model.pth` — best checkpoint by the task's validation metric (highest Top-1 for classification/multi-task, lowest MAE for orientation), containing:
  - `model_state_dict` — model weights
  - `config` — flat dict of training parameters
  - `epoch` — epoch number of best checkpoint
  - `optimizer_state_dict`, `scheduler_state_dict` — optimizer and scheduler state

## Results for the paper models

| Model | Top-1 accuracy | Orientation MAE (ordered) | Epoch |
|-------|---------------|--------------------------|-------|
| rot-aug | 46.72% | — | 80 |
| ori | — | 0.99° | 73 |
| multi-task | 49.52% | 5.32° | 76 |

For context:
- AlexNet trained without rotation augmentation achieves 54.08% Top-1
- A dedicated (non-multitask) orientation model achieves 0.99° MAE
- Random baselines: 90° MAE for ordered [0°, 360°)

The multi-task model maintains near-full classification accuracy (49.52% vs 54.08% for classification-only) while learning meaningful orientation structure in its backbone representations.
