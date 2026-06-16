"""ImageNet dataloader with circular masking and rotation augmentation."""

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from more_anns.preprocessing import (
    CircularMaskTransform,
    DualRandomRotationTransform,
    RandomRotationTransform,
)


class ImageNetDataset(Dataset):
    """Flexible ImageNet dataset supporting single images or rotated pairs.

    Returns a subset of (img1, img2, label, angle1, angle2) based on configuration.
    """

    def __init__(
        self,
        hf_dataset,
        base_transform: transforms.Compose,
        final_transform: transforms.Compose,
        *,
        return_pair: bool = False,
        return_label: bool = True,
        return_angles: bool = False,
        use_rotation_aug: bool = False,
        rotation_quantization: int | None = None,
        interpolation_order: int = 1,
    ):
        """Initialize dataset.

        Args:
            hf_dataset: HuggingFace dataset
            base_transform: Transforms applied before rotation (crop, flip)
            final_transform: Transforms applied after rotation (mask, normalize)
            return_pair: If True, return two rotated versions of the image
            return_label: If True, include class label in output
            return_angles: If True, include rotation angles (requires return_pair=True)
            use_rotation_aug: If True and return_pair=False, apply random rotation augmentation
            rotation_quantization: If set, quantize angles to multiples of this value
            interpolation_order: Interpolation order for rotation (0=nearest, 1=bilinear, 3=bicubic)
        """
        self.dataset = hf_dataset
        self.base_transform = base_transform
        self.final_transform = final_transform
        self.return_pair = return_pair
        self.return_label = return_label
        self.return_angles = return_angles
        self.use_rotation_aug = use_rotation_aug

        if return_angles and not return_pair and not use_rotation_aug:
            msg = "return_angles=True requires return_pair=True or use_rotation_aug=True"
            raise ValueError(msg)

        if return_pair:
            self.rotation_transform = DualRandomRotationTransform(
                quantization=rotation_quantization,
                order=interpolation_order,
            )
        elif use_rotation_aug:
            self.rotation_transform = RandomRotationTransform(
                quantization=rotation_quantization,
                order=interpolation_order,
            )
        else:
            self.rotation_transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = self.base_transform(img)

        if self.rotation_transform is not None:
            imgs, angles = self.rotation_transform(img)
        else:
            imgs = (img,)
            angles = ()
        result = []
        for img in imgs:
            result.append(self.final_transform(img))
        if self.return_label:
            result.append(item["label"])
        if self.return_angles:
            result.extend(angles)
        return tuple(result)


class ImageNetDataModule:
    """Data module for ImageNet with configurable output format.

    Supports three output formats:
    - Single image + label: (img, label) for classification
    - Image pair + angles: (img1, img2, angle1, angle2) for orientation
    - Image pair + label + angles: (img1, img2, label, angle1, angle2) for multitask
    """

    IMAGE_SIZE = 224  # Fixed for AlexNet
    NUM_CLASSES = 1000

    def __init__(
        self,
        dataset: str = "ILSVRC/imagenet-1k",
        batch_size: int = 256,
        num_workers: int = 4,
        *,
        # What to return
        return_pair: bool = False,
        return_label: bool = True,
        return_angles: bool = False,
        # Transforms
        use_circular_mask: bool = True,
        use_rotation_aug: bool = False,
        rotation_quantization: int | None = None,
        interpolation_order: int = 1,
    ):
        """Initialize ImageNet data module.

        Args:
            dataset: HuggingFace dataset name (default: "ILSVRC/imagenet-1k")
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            return_pair: If True, return two rotated versions of each image
            return_label: If True, include class label in output
            return_angles: If True, include rotation angles (requires return_pair=True)
            use_circular_mask: Whether to apply circular masking
            use_rotation_aug: If True and return_pair=False, apply random rotation augmentation
            rotation_quantization: If set, quantize rotation angles to multiples of this value
            interpolation_order: Interpolation order for rotation (0=nearest, 1=bilinear, 3=bicubic)

        Note:
            For ImageNet, you need to accept the terms and login: `huggingface-cli login`
        """
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_pair = return_pair
        self.return_label = return_label
        self.return_angles = return_angles
        self.use_circular_mask = use_circular_mask
        self.use_rotation_aug = use_rotation_aug
        self.rotation_quantization = rotation_quantization
        self.interpolation_order = interpolation_order

        self._train_hf_dataset = None
        self._val_hf_dataset = None

    def _get_base_transform(self, *, train: bool) -> transforms.Compose:
        """Get base transforms (before rotation)."""
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.IMAGE_SIZE),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(self.IMAGE_SIZE),
                ]
            )

    def _get_final_transform(self) -> transforms.Compose:
        """Get final transforms (after rotation)."""
        transform_list = []
        if self.use_circular_mask:
            transform_list.append(CircularMaskTransform(size=self.IMAGE_SIZE, fill_value=128))
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(transform_list)

    def _load_hf_dataset(self, split: str):
        """Load HuggingFace dataset."""
        hf_split = "train" if split == "train" else "validation"
        return load_dataset(self.dataset_name, split=hf_split)

    def _create_dataset(self, *, train: bool) -> ImageNetDataset:
        """Create dataset for train or val split."""
        if train:
            if self._train_hf_dataset is None:
                self._train_hf_dataset = self._load_hf_dataset("train")
            hf_dataset = self._train_hf_dataset
        else:
            if self._val_hf_dataset is None:
                self._val_hf_dataset = self._load_hf_dataset("val")
            hf_dataset = self._val_hf_dataset

        return ImageNetDataset(
            hf_dataset=hf_dataset,
            base_transform=self._get_base_transform(train=train),
            final_transform=self._get_final_transform(),
            return_pair=self.return_pair,
            return_label=self.return_label,
            return_angles=self.return_angles,
            use_rotation_aug=self.use_rotation_aug if train or self.return_angles else False,
            rotation_quantization=self.rotation_quantization,
            interpolation_order=self.interpolation_order,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self._create_dataset(train=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self._create_dataset(train=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )