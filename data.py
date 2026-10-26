"""Dataset loading for the minimal ``area_seg`` reproduction.

Expected layout::

    data_root/
        images/         RGB images
        masks/          ground-truth binary masks
        pseudo_masks/   pre-computed binary pseudo masks

Files are paired by their stem (file name without the extension), so
``images/case_001.jpg``, ``masks/case_001.png`` and
``pseudo_masks/case_001.png`` form one sample.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _index_directory(directory: Path, description: str):
    if not directory.is_dir():
        raise FileNotFoundError(f"{description} directory not found: {directory}")

    index = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in index:
                raise ValueError(
                    f"Duplicate stem '{path.stem}' in {directory}: "
                    f"{index[path.stem].name} and {path.name}"
                )
            index[path.stem] = path

    if not index:
        raise FileNotFoundError(f"No image files found in {directory}")
    return index


def collect_samples(
    data_root,
    image_dir="images",
    mask_dir="masks",
    pseudo_dir="pseudo_masks",
):
    """Collect paired image / mask / pseudo-mask paths.

    Every image must have a real mask and a pseudo mask. Extra mask files
    without a matching image are ignored with a warning.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    images = _index_directory(root / image_dir, "Image")
    masks = _index_directory(root / mask_dir, "Mask")
    pseudo_masks = _index_directory(root / pseudo_dir, "Pseudo-mask")

    missing_masks = sorted(set(images) - set(masks))
    missing_pseudo = sorted(set(images) - set(pseudo_masks))
    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        raise FileNotFoundError(
            f"{len(missing_masks)} image(s) have no real mask, e.g. {preview}"
        )
    if missing_pseudo:
        preview = ", ".join(missing_pseudo[:5])
        raise FileNotFoundError(
            f"{len(missing_pseudo)} image(s) have no pseudo mask, e.g. {preview}"
        )

    extra_masks = sorted(set(masks) - set(images))
    extra_pseudo = sorted(set(pseudo_masks) - set(images))
    if extra_masks:
        print(f"Warning: ignoring {len(extra_masks)} mask file(s) without an image.")
    if extra_pseudo:
        print(f"Warning: ignoring {len(extra_pseudo)} pseudo-mask file(s) without an image.")

    stems = sorted(images)
    return [(images[stem], masks[stem], pseudo_masks[stem]) for stem in stems]


def split_indices(num_samples, val_ratio, test_ratio, seed):
    """Split sample indices into an AL pool, a validation set and a test set."""
    if num_samples < 3:
        raise ValueError("At least 3 paired samples are required.")
    if val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("--val-ratio and --test-ratio must both be greater than 0.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be smaller than 1.")

    rng = np.random.RandomState(seed)
    order = rng.permutation(num_samples)

    num_val = max(int(round(num_samples * val_ratio)), 1)
    num_test = max(int(round(num_samples * test_ratio)), 1)
    if num_val + num_test >= num_samples:
        raise ValueError(
            "Not enough samples for the requested validation/test split. "
            f"num_samples={num_samples}, num_val={num_val}, num_test={num_test}"
        )

    val_indices = np.sort(order[:num_val])
    test_indices = np.sort(order[num_val:num_val + num_test])
    pool_indices = np.sort(order[num_val + num_test:])
    return pool_indices, val_indices, test_indices


class SegmentationDataset(Dataset):
    """Returns ``(image, real_mask, pseudo_mask, global_index)``."""

    def __init__(self, samples, img_size=224, augment=False):
        self.samples = samples
        self.img_size = int(img_size)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, pseudo_path = self.samples[index]
        size = (self.img_size, self.img_size)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB").resize(size, Image.BILINEAR)
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L").resize(size, Image.NEAREST)
        with Image.open(pseudo_path) as pseudo_file:
            pseudo = pseudo_file.convert("L").resize(size, Image.NEAREST)

        image = TF.to_tensor(image)
        mask = (TF.to_tensor(mask) > 0.5).float()
        pseudo = (TF.to_tensor(pseudo) > 0.5).float()

        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
                pseudo = torch.flip(pseudo, dims=[2])
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
                pseudo = torch.flip(pseudo, dims=[1])

        return image, mask, pseudo, index
