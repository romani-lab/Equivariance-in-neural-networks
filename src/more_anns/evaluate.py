"""Post-hoc evaluation of orientation checkpoints.

Subcommands:
    stats   — Prediction distribution stats + heatmap over the validation set.
    sweep   — Rotate a single image through angles and plot predicted vs true.
"""

import logging
from pathlib import Path

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image as PILImage
from skimage import transform as sktransform
from torchvision import transforms
from tqdm import tqdm

from more_anns.data import ImageNetDataModule
from more_anns.losses import get_angle_representation, get_orientation_loss
from more_anns.metrics import orientation_mae
from more_anns.models import AlexNet
from more_anns.preprocessing import CircularMaskTransform

mpl.use("Agg")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(path: str):
    """Load checkpoint and return (ckpt dict, config dict, device str)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ckpt, cfg, device


def _build_model(cfg, rep):
    """Reconstruct AlexNet from checkpoint config."""
    model_type = cfg.get("model_type", "siamese")
    is_multitask = "classification_weight" in cfg or "orientation_weight" in cfg
    if model_type == "siamese":
        task = "multitask" if is_multitask else "orientation"
        model = AlexNet(
            task=task,
            orientation_fusion=cfg.get("orientation_fusion", "concat"),
            similarity_layer=cfg.get("similarity_layer", "classifier.5"),
            orientation_output_dim=rep.output_dim,
            orientation_activation=cfg.get("orientation_activation", "linear"),
        )
    elif model_type == "absolute":
        model = AlexNet(
            task="absolute_orientation",
            orientation_output_dim=rep.output_dim,
            orientation_activation=cfg.get("orientation_activation", "linear"),
        )
    else:
        msg = f"Unknown model_type: {model_type}"
        raise ValueError(msg)
    return model, model_type, is_multitask


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Evaluate orientation checkpoints."""


# ---------------------------------------------------------------------------
# stats subcommand (formerly the top-level evaluate command)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--max-batches", default=50, type=int, help="Max val batches (0=all)")
def stats(checkpoint: str, max_batches: int) -> None:
    """Prediction distribution stats + heatmap over the validation set."""
    ckpt, cfg, device = _load_checkpoint(checkpoint)
    log.info(f"Config: {cfg}")

    rep = get_angle_representation(cfg["angle_representation"])
    criterion = get_orientation_loss(cfg["angle_representation"], cfg.get("loss_type", "mse"))

    model, model_type, is_multitask = _build_model(cfg, rep)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    use_pair = model_type != "absolute"
    data_module = ImageNetDataModule(
        dataset=cfg["dataset"],
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 4),
        return_pair=use_pair,
        return_label=is_multitask,
        return_angles=True,
        use_rotation_aug=not use_pair,
        use_circular_mask=cfg.get("use_circular_mask", True),
        rotation_quantization=cfg.get("rotation_quantization"),
        interpolation_order=cfg.get("interpolation_order", 1),
    )
    val_loader = data_module.val_dataloader()

    all_pred_angles = []
    all_target_angles = []
    all_raw_preds = []
    total_loss = 0.0
    total_mae = 0.0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            if max_batches and i >= max_batches:
                break

            if model_type == "absolute":
                img, angle = batch
                img = img.to(device)
                angle1 = torch.zeros_like(angle).to(device, dtype=torch.float32)
                angle2 = angle.to(device, dtype=torch.float32)
                (pred,) = model(img)
            elif is_multitask:
                img1, img2, _label, angle1, angle2 = batch
                img1 = img1.to(device)
                img2 = img2.to(device)
                angle1 = angle1.to(device, dtype=torch.float32)
                angle2 = angle2.to(device, dtype=torch.float32)
                _logits1, _logits2, pred = model(img1, img2)
            else:
                img1, img2, angle1, angle2 = batch
                img1 = img1.to(device)
                img2 = img2.to(device)
                angle1 = angle1.to(device, dtype=torch.float32)
                angle2 = angle2.to(device, dtype=torch.float32)
                (pred,) = model(img1, img2)

            loss = criterion(pred, angle1, angle2)
            mae = orientation_mae(pred, angle1, angle2, rep, ordered=False)

            pred_deg = rep.decode(pred)
            if rep.ordered:
                target_deg = (angle2 - angle1) % 360
            else:
                delta = torch.abs(angle2 - angle1)
                target_deg = torch.min(delta, 360 - delta)

            all_pred_angles.append(pred_deg.cpu())
            all_target_angles.append(target_deg.cpu())
            all_raw_preds.append(pred.cpu())

            bs = pred.size(0)
            total_loss += loss.item() * bs
            total_mae += mae.item() * bs
            total += bs

    all_pred = torch.cat(all_pred_angles)
    all_target = torch.cat(all_target_angles)
    all_raw = torch.cat(all_raw_preds)

    log.info("=" * 60)
    log.info(f"Checkpoint: {checkpoint}")
    log.info(f"Epoch: {ckpt['epoch']}")
    log.info(f"Samples evaluated: {total}")
    log.info(f"Loss: {total_loss / total:.4f}")
    log.info(f"MAE (unordered): {total_mae / total:.2f}")
    log.info("Predicted angles (decoded):")
    log.info(f"  mean: {all_pred.mean():.1f}, std: {all_pred.std():.1f}")
    log.info(f"  min:  {all_pred.min():.1f}, max: {all_pred.max():.1f}")
    log.info(f"  median: {all_pred.median():.1f}")
    log.info("Target angles:")
    log.info(f"  mean: {all_target.mean():.1f}, std: {all_target.std():.1f}")
    log.info(f"  min:  {all_target.min():.1f}, max: {all_target.max():.1f}")
    log.info("Raw model outputs:")
    for d in range(all_raw.shape[1]):
        col = all_raw[:, d]
        log.info(f"  dim {d}: mean={col.mean():.4f}, std={col.std():.4f}, min={col.min():.4f}, max={col.max():.4f}")
    log.info("=" * 60)

    # Save prediction heatmap
    max_angle = 360.0 if rep.ordered else 180.0
    n_bins = 36 if rep.ordered else 18
    bin_size = max_angle / n_bins

    pred_bins = (all_pred / bin_size).long().clamp(0, n_bins - 1)
    target_bins = (all_target / bin_size).long().clamp(0, n_bins - 1)
    indices = pred_bins * n_bins + target_bins
    heatmap = torch.bincount(indices, minlength=n_bins * n_bins).float().view(n_bins, n_bins)
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
    ax.set_title(f"Epoch {ckpt['epoch']} \u2014 MAE {total_mae / total:.1f}")
    fig.colorbar(im, ax=ax, label="P(pred | target)")
    fig.tight_layout()

    out_path = Path(checkpoint).parent / "heatmap.png"
    fig.savefig(out_path, dpi=150)
    log.info(f"Heatmap saved to {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# sweep subcommand
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--image-index", default=0, type=int, help="Index into ImageNet validation set")
@click.option("--num-angles", default=12, type=int, help="Number of angles to sweep (evenly spaced in [0, 360))")
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output path (default: rotation_sweep.png next to checkpoint)",
)
def sweep(checkpoint: str, image_index: int, num_angles: int, output: str | None) -> None:
    """Rotate a single image through all angle pairs and plot prediction matrix.

    For siamese models: shows an NxN matrix where cell (x, y) is the predicted
    relative rotation between image 1 at angle x and image 2 at angle y.
    Thumbnail images are shown alongside degree labels on both axes.
    """
    ckpt, cfg, device = _load_checkpoint(checkpoint)
    log.info(f"Config: {cfg}")

    rep = get_angle_representation(cfg["angle_representation"])

    model, model_type, _is_multitask = _build_model(cfg, rep)
    if model_type == "absolute":
        msg = "sweep matrix requires a siamese model (not absolute)"
        raise click.UsageError(msg)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load a single image from ImageNet validation
    log.info(f"Loading image {image_index} from validation set...")
    ds = load_dataset(cfg["dataset"], split="validation")
    pil_img = ds[image_index]["image"]
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    base_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    pil_img = base_transform(pil_img)

    use_circular_mask = cfg.get("use_circular_mask", True)
    interpolation_order = cfg.get("interpolation_order", 1)
    mask_tfm = CircularMaskTransform(size=224, fill_value=128) if use_circular_mask else None
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Generate rotated images
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    thumb_size = 48
    thumbnails = []
    model_inputs = []

    log.info(f"Rotating image to {num_angles} angles...")
    for angle in angles:
        arr = np.array(pil_img)
        rotated = sktransform.rotate(
            arr,
            angle,
            mode="constant",
            cval=128,
            preserve_range=True,
            order=interpolation_order,
        )
        rotated_pil = PILImage.fromarray(rotated.astype(np.uint8))
        if mask_tfm:
            rotated_pil = mask_tfm(rotated_pil)

        thumb = rotated_pil.resize((thumb_size, thumb_size), PILImage.LANCZOS)
        thumbnails.append(np.array(thumb))
        model_inputs.append(to_tensor(rotated_pil))

    # Predict all pairs — batched per img1 column
    predictions = np.zeros((num_angles, num_angles))
    img2_batch = torch.stack(model_inputs).to(device)

    log.info(f"Running {num_angles}x{num_angles} = {num_angles**2} predictions...")
    with torch.no_grad():
        for x_idx in range(num_angles):
            img1_batch = model_inputs[x_idx].unsqueeze(0).expand_as(img2_batch).to(device)
            outputs = model(img1_batch, img2_batch)
            pred = outputs[-1]
            predictions[:, x_idx] = rep.decode(pred).cpu().numpy()

    # --- Plot with exact positioning for square thumbnails ---
    cell_in = 0.55  # inches per heatmap cell
    heatmap_in = cell_in * num_angles
    gap_in = 0.08
    wheel_in = 1.6
    margin_l, margin_r = 0.2, 0.4
    margin_b, margin_t = 0.3, 0.9

    fig_w = margin_l + cell_in + gap_in + heatmap_in + gap_in + wheel_in + margin_r
    fig_h = margin_b + cell_in + gap_in + heatmap_in + margin_t

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Convert inches to figure fractions
    hm_left = (margin_l + cell_in + gap_in) / fig_w
    hm_bottom = (margin_b + cell_in + gap_in) / fig_h
    hm_w = heatmap_in / fig_w
    hm_h = heatmap_in / fig_h

    ax_main = fig.add_axes([hm_left, hm_bottom, hm_w, hm_h])

    # Thumbnail strips — each thumbnail is cell_in x cell_in inches
    ax_x = fig.add_axes([hm_left, margin_b / fig_h, hm_w, cell_in / fig_h])
    ax_y = fig.add_axes([margin_l / fig_w, hm_bottom, cell_in / fig_w, hm_h])

    # Color wheel
    wh_left = (margin_l + cell_in + gap_in + heatmap_in + gap_in) / fig_w
    wh_bottom = hm_bottom + (hm_h - wheel_in / fig_h) / 2  # vertically centered
    ax_wheel = fig.add_axes(
        [wh_left, wh_bottom, wheel_in / fig_w, wheel_in / fig_h],
        projection="polar",
    )

    # Heatmap with rainbow colormap (always 0-360° so colors are consistent)
    ax_main.imshow(
        predictions,
        cmap="hsv",
        vmin=0,
        vmax=360,
        origin="lower",
        aspect="equal",
    )

    # Degree tick labels on top (img1) and right (img2)
    ax_main.set_xticks(range(num_angles))
    ax_main.set_xticklabels([f"{a:.0f}" for a in angles], fontsize=7)
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("Image 1 rotation")

    ax_main.set_yticks(range(num_angles))
    ax_main.set_yticklabels([f"{a:.0f}" for a in angles], fontsize=7)
    ax_main.yaxis.tick_right()

    # X-axis image strip (bottom): image 1 thumbnails
    x_strip = np.concatenate(thumbnails, axis=1)
    ax_x.imshow(x_strip, extent=[-0.5, num_angles - 0.5, 0, 1], aspect="auto")
    ax_x.set_xlim(-0.5, num_angles - 0.5)
    ax_x.axis("off")

    # Y-axis image strip (left): image 2 thumbnails (reversed so 0° at bottom)
    y_strip = np.concatenate(thumbnails[::-1], axis=0)
    ax_y.imshow(y_strip, extent=[0, 1, -0.5, num_angles - 0.5], aspect="auto")
    ax_y.set_ylim(-0.5, num_angles - 0.5)
    ax_y.axis("off")

    # Color wheel legend (always full 360° circle for consistent color mapping)
    n_seg = 256
    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    seg_width = 2 * np.pi / n_seg
    wheel_colors = plt.cm.hsv(np.linspace(0, 1, n_seg))
    ax_wheel.bar(theta, 1, width=seg_width, bottom=0.5, color=wheel_colors, edgecolor="none")
    ax_wheel.set_ylim(0, 1.7)
    ax_wheel.set_yticks([])
    ax_wheel.set_xticks(np.arange(0, 360, 30) * np.pi / 180)
    ax_wheel.set_xticklabels([f"{a:.0f}" for a in np.arange(0, 360, 30)], fontsize=8)
    ax_wheel.set_theta_zero_location("N")
    ax_wheel.set_theta_direction(-1)
    ax_wheel.set_title("Predicted\nangle", fontsize=9, pad=12)
    ax_wheel.spines["polar"].set_visible(False)

    fig.text(
        0.5,
        0.98,
        f"Rotation sweep \u2014 image {image_index}, epoch {ckpt['epoch']}",
        ha="center",
        va="top",
        fontsize=11,
    )

    out_path = output or str(Path(checkpoint).parent / "rotation_sweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Summary: compare predictions to true relative rotations
    true_matrix = np.zeros((num_angles, num_angles))
    for yi, ay in enumerate(angles):
        for xi, ax_val in enumerate(angles):
            delta = abs(ay - ax_val)
            if rep.ordered:
                true_matrix[yi, xi] = (ay - ax_val) % 360
            else:
                true_matrix[yi, xi] = min(delta, 360 - delta)

    errors = np.abs(predictions - true_matrix)
    if not rep.ordered:
        errors = np.minimum(errors, 180 - errors)

    log.info(f"Rotation sweep (image {image_index}):")
    log.info(f"  MAE: {errors.mean():.1f}")
    log.info(f"  Predictions range: [{predictions.min():.1f}, {predictions.max():.1f}]")
    log.info(f"  Saved to {out_path}")