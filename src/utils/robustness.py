import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms
from tqdm import tqdm


IMAGENET_BASE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

IMAGENET_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

DEFAULT_ANGLES = np.arange(0, 360, 24)


def rotate_pil(img, angle):
    """
    Rotate a PIL image by angle with bilinear interpolation and black fill.

    Parameters
    ----------
    img : PIL.Image.Image
        image to rotate
    angle : float
        rotation angle in degrees

    Returns
    -------
    rotated_img : PIL.Image.Image
        rotated image, or the original image when angle is 0
    """
    if angle == 0:
        return img
    return img.rotate(-angle, resample=PILImage.BILINEAR, fillcolor=(0, 0, 0))


def circular_error_ordered(estimated, true_angle):
    """
    Compute circular-wrapped absolute error for angles in [0, 360).

    Parameters
    ----------
    estimated : np.ndarray or float
        estimated angle or angles in degrees
    true_angle : np.ndarray or float
        true angle or angles in degrees

    Returns
    -------
    error : np.ndarray or float
        absolute circular error in degrees
    """
    diff = np.abs(estimated - true_angle)
    return np.minimum(diff, 360.0 - diff)


def load_imagenet_samples(cache_file, n_images=100, seed=42):
    """Load cached ImageNet validation samples, or download and cache them.

    Parameters
    ----------
    cache_file : str or Path
        Path to the pickle cache file.
    n_images : int
        Number of images to sample.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    samples : list of dict
        Each dict has 'image' (PIL) and 'label' (int).
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            samples = pickle.load(f)
        print(f'Loaded {len(samples)} cached images from {cache_file}')
        return samples

    from datasets import load_dataset

    ds = load_dataset('ILSVRC/imagenet-1k', split='validation', streaming=True)
    rng = np.random.default_rng(seed)
    N_VAL = 50000
    indices = sorted(rng.choice(N_VAL, size=n_images, replace=False))

    samples = []
    idx_set = set(indices)
    for i, item in enumerate(tqdm(ds, total=indices[-1] + 1, desc='Loading images')):
        if i in idx_set:
            samples.append({'image': item['image'], 'label': item['label']})
        if i > indices[-1]:
            break

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(samples, f)
    print(f'Downloaded and cached {len(samples)} images to {cache_file}')
    return samples


def evaluate_rotation_robustness(model, samples, angles=None, device='cpu',
                                 use_backbone=False):
    """Evaluate top-1 classification accuracy at each rotation angle.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate. Must return (logits,) or have .backbone/.classifier_head.
    samples : list of dict
        ImageNet samples from load_imagenet_samples.
    angles : array-like or None
        Rotation angles in degrees. Defaults to DEFAULT_ANGLES.
    device : str
        Torch device.
    use_backbone : bool
        If True, use model.backbone + model.classifier_head instead of model().

    Returns
    -------
    correct : np.ndarray, shape (n_images, n_angles)
        Boolean array of correct predictions.
    """
    if angles is None:
        angles = DEFAULT_ANGLES
    angles = np.asarray(angles)
    n_images = len(samples)
    correct = np.zeros((n_images, len(angles)), dtype=bool)

    with torch.no_grad():
        for i, item in enumerate(tqdm(samples, desc='Evaluating rotation')):
            label = item['label']
            img = item['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = IMAGENET_BASE_TRANSFORM(img)

            batch = []
            for angle in angles:
                rotated = rotate_pil(img, angle)
                batch.append(IMAGENET_TO_TENSOR(rotated))

            batch = torch.stack(batch).to(device)
            if use_backbone:
                emb = model.backbone(batch)
                logits = model.classifier_head(emb)
            else:
                (logits,) = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            correct[i] = (preds == label)

    return correct


def evaluate_orientation_robustness(model, samples, rep, angles=None,
                                    device='cpu', mode='orientation'):
    """Evaluate orientation estimation for image pairs (0 deg ref vs rotated).

    Parameters
    ----------
    model : torch.nn.Module
        Siamese orientation model.
    samples : list of dict
        ImageNet samples from load_imagenet_samples.
    rep : AngleRepresentation
        Angle representation object with .decode() method.
    angles : array-like or None
        Rotation angles in degrees. Defaults to DEFAULT_ANGLES.
    device : str
        Torch device.
    mode : str
        'orientation' means model returns (orient_pred,);
        'multitask' means model returns (logits1, logits2, orient_pred)

    Returns
    -------
    pred_angles : np.ndarray, shape (n_images, n_angles)
        Predicted angles in degrees.
    """
    if angles is None:
        angles = DEFAULT_ANGLES
    angles = np.asarray(angles)
    n_images = len(samples)

    img_tensors = []
    for item in tqdm(samples, desc='Preprocessing'):
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = IMAGENET_BASE_TRANSFORM(img)
        tensors = [IMAGENET_TO_TENSOR(rotate_pil(img, a)) for a in angles]
        img_tensors.append(tensors)

    pred_angles = np.zeros((n_images, len(angles)))

    with torch.no_grad():
        for i in tqdm(range(n_images), desc='Evaluating orientation'):
            img1_tensor = img_tensors[i][0]
            img1_batch = img1_tensor.unsqueeze(0).expand(len(angles), -1, -1, -1).to(device)
            img2_batch = torch.stack([img_tensors[i][j] for j in range(len(angles))]).to(device)

            if mode == 'orientation':
                (orient_pred,) = model(img1_batch, img2_batch)
            elif mode == 'multitask':
                _logits1, _logits2, orient_pred = model(img1_batch, img2_batch)
            else:
                raise ValueError(f'Unknown mode: {mode}')

            pred_deg = rep.decode(orient_pred).cpu().numpy()
            pred_angles[i] = pred_deg

    return pred_angles
