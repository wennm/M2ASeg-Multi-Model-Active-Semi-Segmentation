"""Training and evaluation helpers for the minimal reproduction.

These follow the original segmentation strategy: per-sample Dice loss, SGD
with momentum and weight decay, cosine learning-rate schedule, validation
Dice-based early stopping, and logits thresholded at zero for prediction.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DiceLoss(nn.Module):
    """Dice loss used by the original experiments (with smoothing = 1.0)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


def _batch_targets(real_masks, pseudo_masks, indices, pseudo_indices):
    use_pseudo = torch.as_tensor(
        [index.item() in pseudo_indices for index in indices],
        dtype=torch.bool,
        device=real_masks.device,
    ).view(-1, 1, 1, 1)
    return torch.where(use_pseudo, pseudo_masks, real_masks)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    labeled_indices,
    pseudo_indices,
    device,
):
    model.train()
    loss_sum = 0.0
    num_samples = 0

    for images, real_masks, pseudo_masks, indices in loader:
        for index in indices:
            index = int(index)
            if index not in labeled_indices and index not in pseudo_indices:
                raise RuntimeError(f"Sample {index} is neither labeled nor pseudo-labeled.")

        images = images.to(device)
        real_masks = real_masks.to(device)
        pseudo_masks = pseudo_masks.to(device)
        targets = _batch_targets(real_masks, pseudo_masks, indices, pseudo_indices)

        outputs = model(images)
        # The original implementation averages per-sample Dice losses.
        sample_losses = torch.stack(
            [criterion(outputs[i:i + 1], targets[i:i + 1]) for i in range(outputs.size(0))]
        )
        loss = sample_losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        num_samples += images.size(0)

    return loss_sum / max(num_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return mean Dice and mean IoU on a data split."""
    model.eval()
    dice_values = []
    iou_values = []

    for images, real_masks, _, _ in loader:
        images = images.to(device)
        targets = real_masks.to(device)
        predictions = (model(images) >= 0).float()

        for prediction, target in zip(predictions, targets):
            pred_sum = prediction.sum()
            target_sum = target.sum()
            intersection = (prediction * target).sum()

            if pred_sum.item() == 0 and target_sum.item() == 0:
                dice_values.append(1.0)
            else:
                dice_values.append(
                    float((2.0 * intersection / (pred_sum + target_sum)).item())
                )

            union = pred_sum + target_sum - intersection
            iou_values.append(
                1.0 if union.item() == 0 else float((intersection / union).item())
            )

    if not dice_values:
        raise ValueError("Cannot evaluate on an empty data split.")
    return float(np.mean(dice_values)), float(np.mean(iou_values))


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    momentum,
    weight_decay,
    patience,
    labeled_indices,
    pseudo_indices,
    device,
    verbose=True,
):
    """Train for a stage and restore the best-validation checkpoint.

    The model is updated in place and is intentionally reused across active
    learning rounds, as in the original code.
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = DiceLoss()

    best_val_dice = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            labeled_indices,
            pseudo_indices,
            device,
        )
        val_dice, _ = evaluate(model, val_loader, device)
        scheduler.step()
        completed_epochs = epoch

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"    epoch {epoch:3d}/{epochs} | "
                f"train_loss {train_loss:.4f} | val_dice {val_dice:.4f}"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print(f"    early stopping after {epoch} epoch(s)")
            break

    model.load_state_dict(best_state)
    return best_val_dice, completed_epochs
