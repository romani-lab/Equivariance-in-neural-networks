"""Data preprocessing utilities for circular masking and rotations."""

import numpy as np
from PIL import Image
from skimage import transform as sktransform


def create_circular_mask(size: int) -> np.ndarray:
    """Create a circular mask for square images.

    - Creates a mask where pixels inside the circle are 1, outside are 0
    - Circle is centered and has radius size/2, i.e. extends to the edge.

    Vectorized equivalent of this MATLAB code::

        sz = size(I0,1);
        mask = zeros(sz, sz);
        center = sz / 2;
        for i = 1:sz
            for j = 1:sz
                distance = sqrt((i - center)^2 + (j - center)^2);
                if distance <= center
                    mask(i, j) = 1;
                end
            end
        end

    Args:
        size: Size of the square mask (both width and height)

    Returns:
        Binary mask array of shape (size, size) with 1s inside circle, 0s outside
    """
    center = size / 2.0
    y, x = np.ogrid[0.5 : size + 0.5, 0.5 : size + 0.5]
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = (distance <= center).astype(np.float32)
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray, fill_value: int = 128) -> np.ndarray:
    """Apply circular mask to an image, filling outside pixels with fill_value.

    Equivalent to MATLAB:
    Icirc = uint8(128 * ones(sz));
    Icirc(mask == 1) = I0(mask == 1);

    Args:
        image: Input image array of shape (H, W) or (H, W, C)
        mask: Binary mask of shape (H, W)
        fill_value: Value to fill masked-out pixels (default: 128 for gray)

    Returns:
        Masked image with same shape as input
    """
    # Create output filled with fill_value
    masked = np.full_like(image, fill_value)

    # Copy pixels where mask is 1
    if image.ndim == 2:  # noqa: PLR2004
        # Grayscale
        masked[mask == 1] = image[mask == 1]
    elif image.ndim == 3:  # noqa: PLR2004
        # RGB: expand mask to all channels
        mask_3d = mask[:, :, np.newaxis]
        masked = np.where(mask_3d == 1, image, masked)

    return masked


class CircularMaskTransform:
    """Transform to apply circular masking to PIL Images."""

    def __init__(self, size: int, fill_value: int = 128):
        """Initialize circular mask transform.

        Args:
            size: Size of the square image
            fill_value: Value to fill masked-out pixels
        """
        self.size = size
        self.fill_value = fill_value
        self.mask = create_circular_mask(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply circular mask to PIL Image.

        Args:
            img: Input PIL Image

        Returns:
            Masked PIL Image
        """
        # Convert to numpy
        img_array = np.array(img)

        # Apply mask
        masked_array = apply_mask(img_array, self.mask, self.fill_value)

        # Convert back to PIL
        return Image.fromarray(masked_array.astype(np.uint8))


class RandomRotationTransform:
    """Transform to apply random rotation using skimage.transform.rotate."""

    def __init__(self, quantization: int | None = None, order: int = 1):
        """Initialize random rotation transform.

        Args:
            quantization: If set, quantize angles to multiples of this value.
                E.g., quantization=24 gives angles 0, 24, 48, ..., 336 (15 values).
            order: Interpolation order for skimage.transform.rotate (0=nearest, 1=bilinear, 3=bicubic)
        """
        self.quantization = quantization
        self.order = order

    def __call__(self, img: Image.Image) -> tuple[tuple[Image.Image], tuple[float]]:
        """Apply random rotation to PIL Image.

        Args:
            img: Input PIL Image

        Returns:
            Tuple of ((rotated_img,), (angle,))
        """
        if self.quantization is not None:
            # Select from quantized angles (e.g., 0, 24, 48, ..., 336 for quantization=24)
            num_angles = 360 // self.quantization
            angle = float(np.random.randint(0, num_angles) * self.quantization)
        else:
            # Random angle in [0, 360)
            angle = np.random.uniform(0, 360)

        # Convert to numpy
        img_array = np.array(img)

        # Rotate using skimage (handles multi-channel images)
        rotated = sktransform.rotate(img_array, angle, mode="constant", cval=128, preserve_range=True, order=self.order)

        # Convert back to PIL
        rotated_img = Image.fromarray(rotated.astype(np.uint8))

        return (rotated_img,), (angle,)


class DualRandomRotationTransform:
    """Transform that returns two random rotations and their angle difference.

    Used for Siamese network training with orientation regression.
    """

    def __init__(self, quantization: int | None = None, order: int = 1):
        """Initialize dual random rotation transform.

        Args:
            quantization: If set, quantize angles to multiples of this value.
                E.g., quantization=24 gives angles 0, 24, 48, ..., 336 (15 values).
            order: Interpolation order for skimage.transform.rotate (0=nearest, 1=bilinear, 3=bicubic)
        """
        self.quantization = quantization
        self.order = order

    def __call__(self, img: Image.Image) -> tuple[Image.Image, Image.Image, float, float]:
        """Apply two random rotations and return both angles.

        Args:
            img: Input PIL Image

        Returns:
            Tuple of (rotated_img1, rotated_img2, angle1, angle2)
            where angles are in degrees [0, 360)
        """
        if self.quantization is not None:
            # Select from quantized angles
            num_angles = 360 // self.quantization
            angle1 = float(np.random.randint(0, num_angles) * self.quantization)
            angle2 = float(np.random.randint(0, num_angles) * self.quantization)
        else:
            # Two random angles
            angle1 = np.random.uniform(0, 360)
            angle2 = np.random.uniform(0, 360)

        # Convert to numpy
        img_array = np.array(img)

        # Rotate twice
        rotated1 = sktransform.rotate(
            img_array, angle1, mode="constant", cval=128, preserve_range=True, order=self.order
        )
        rotated2 = sktransform.rotate(
            img_array, angle2, mode="constant", cval=128, preserve_range=True, order=self.order
        )

        # Convert to PIL
        img1 = Image.fromarray(rotated1.astype(np.uint8))
        img2 = Image.fromarray(rotated2.astype(np.uint8))

        return (img1, img2), (angle1, angle2)
