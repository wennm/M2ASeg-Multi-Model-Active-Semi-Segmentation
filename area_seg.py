"""The ``area_seg`` sampling strategy.

For every training-pool image, the score is the number of foreground pixels in
its pre-computed pseudo mask. Masks are resized to the training resolution with
nearest-neighbour interpolation first, matching the original implementation,
which computed ``(pseudo_mask > 0.5).sum()`` on the resized tensor.

The same score is used for the initial labeled set and for every active
learning round. The real ground-truth mask of a selected image is used for
training; the pseudo mask is only used for ranking (and optionally for SSL).
"""

import numpy as np
from PIL import Image


def compute_pseudo_areas(samples, img_size):
    """Return the pseudo-mask foreground area of every sample.

    ``samples`` is the list returned by :func:`data.collect_samples`, i.e.
    ``[(image_path, mask_path, pseudo_mask_path), ...]``.
    """
    size = (int(img_size), int(img_size))
    areas = np.empty(len(samples), dtype=np.int64)
    for index, (_, _, pseudo_path) in enumerate(samples):
        with Image.open(pseudo_path) as pseudo_file:
            pseudo = pseudo_file.convert("L").resize(size, Image.NEAREST)
        # Equivalent to (tensor > 0.5) for an 8-bit mask.
        areas[index] = int((np.asarray(pseudo) > 127).sum())
    return areas


def select_largest_area(candidate_indices, areas, n):
    """Select ``n`` candidates with the largest pseudo-mask area.

    Returns the selected indices in descending area order. Ties are broken by
    the lower pool index (stable sort).
    """
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.size == 0:
        return candidates
    n = min(int(n), candidates.size)
    order = np.argsort(-areas[candidates], kind="stable")
    return candidates[order[:n]]
