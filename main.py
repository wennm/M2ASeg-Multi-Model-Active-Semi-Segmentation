"""Minimal reproduction of the ``area_seg`` active learning strategy.

Workflow:
1. Split the paired images into an active-learning pool, a validation set and
   a test set.
2. Select the initial labeled set by largest pseudo-mask area.
3. Train U-Net on the labeled set (real masks).
4. Each round: select the remaining pool images with the largest pseudo-mask
   areas, add them to the labeled set (real masks), and retrain.
5. Optionally (``--use-ssl``) train once more with pseudo masks as labels for
   every remaining pool image.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from area_seg import compute_pseudo_areas, select_largest_area
from data import SegmentationDataset, collect_samples, split_indices
from model import Unet
from train import evaluate, train_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Minimal area_seg active learning reproduction (U-Net only)."
    )
    parser.add_argument("--data-root", required=True, help="Root directory of the dataset.")
    parser.add_argument("--image-dir", default="images", help="Image sub-directory.")
    parser.add_argument("--mask-dir", default="masks", help="Ground-truth mask sub-directory.")
    parser.add_argument(
        "--pseudo-dir", default="pseudo_masks", help="Pre-computed pseudo-mask sub-directory."
    )

    parser.add_argument("--img-size", type=int, default=224, help="Square training resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")

    parser.add_argument(
        "--init-size", type=int, default=15, help="Number of initially labeled images."
    )
    parser.add_argument(
        "--query-size", type=int, default=15, help="Number of images selected per AL round."
    )
    parser.add_argument("--rounds", type=int, default=4, help="Number of AL rounds.")

    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per stage.")
    parser.add_argument(
        "--ssl-epochs",
        type=int,
        default=None,
        help="Max epochs for the optional SSL stage (default: --epochs).",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Initial SGD learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="SGD weight decay.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience on validation Dice; 0 disables it.",
    )

    parser.add_argument(
        "--use-ssl",
        action="store_true",
        help="After AL, train once with pre-computed pseudo masks for the remaining pool.",
    )
    parser.add_argument("--no-aug", action="store_true", help="Disable random flip augmentation.")
    parser.add_argument(
        "--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'."
    )
    parser.add_argument("--out-dir", default="outputs", help="Directory for metrics and logs.")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu.")
    return device


def validate_args(parser, args):
    if args.img_size <= 0:
        parser.error("--img-size must be positive.")
    if args.val_ratio <= 0 or args.test_ratio <= 0:
        parser.error("--val-ratio and --test-ratio must both be greater than 0.")
    if args.val_ratio + args.test_ratio >= 1:
        parser.error("--val-ratio + --test-ratio must be smaller than 1.")
    if args.init_size <= 0:
        parser.error("--init-size must be positive.")
    if args.rounds < 0:
        parser.error("--rounds must be >= 0.")
    if args.rounds > 0 and args.query_size <= 0:
        parser.error("--query-size must be positive when --rounds > 0.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.ssl_epochs is not None and args.ssl_epochs <= 0:
        parser.error("--ssl-epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    samples = collect_samples(
        args.data_root,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        pseudo_dir=args.pseudo_dir,
    )
    print(f"Paired samples: {len(samples)}")

    pool_indices, val_indices, test_indices = split_indices(
        len(samples), args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"AL pool: {len(pool_indices)} | validation: {len(val_indices)} | "
        f"test: {len(test_indices)}"
    )

    if args.init_size > len(pool_indices):
        raise ValueError(
            f"--init-size ({args.init_size}) exceeds the AL pool size "
            f"({len(pool_indices)}). Reduce --init-size or the split ratios."
        )

    print("Computing pseudo-mask areas...")
    areas = compute_pseudo_areas(samples, args.img_size)

    train_dataset = SegmentationDataset(
        samples, img_size=args.img_size, augment=not args.no_aug
    )
    eval_dataset = SegmentationDataset(samples, img_size=args.img_size, augment=False)

    pin_memory = device.type == "cuda"
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = Unet(in_ch=3, out_ch=1).to(device)
    ssl_epochs = args.epochs if args.ssl_epochs is None else args.ssl_epochs

    run_dir = Path(args.out_dir) / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    selections_path = run_dir / "selected_samples.csv"
    args_path = run_dir / "args.json"

    with args_path.open("w", encoding="utf-8") as args_file:
        json.dump(vars(args), args_file, indent=2, ensure_ascii=False)

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file, \
            selections_path.open("w", newline="", encoding="utf-8") as selections_file:
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "stage",
                "round",
                "n_labeled",
                "n_pseudo",
                "val_dice",
                "test_dice",
                "test_iou",
                "epochs",
                "seconds",
            ],
        )
        selections_writer = csv.DictWriter(
            selections_file,
            fieldnames=[
                "stage",
                "round",
                "selection_order",
                "index",
                "image",
                "pseudo_area",
            ],
        )
        metrics_writer.writeheader()
        selections_writer.writeheader()

        def record_selection(stage, round_index, indices):
            for order, index in enumerate(indices, start=1):
                index = int(index)
                selections_writer.writerow(
                    {
                        "stage": stage,
                        "round": round_index,
                        "selection_order": order,
                        "index": index,
                        "image": samples[index][0].name,
                        "pseudo_area": int(areas[index]),
                    }
                )
            selections_file.flush()

        def run_stage(stage, round_index, labeled_indices, pseudo_indices, epochs):
            train_indices = sorted(labeled_indices | pseudo_indices)
            generator = torch.Generator()
            generator.manual_seed(args.seed + 10_000 + round_index)
            train_loader = DataLoader(
                Subset(train_dataset, train_indices),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                generator=generator,
            )

            print(
                f"\n[{stage}] real labels: {len(labeled_indices)} | "
                f"pseudo labels: {len(pseudo_indices)} | training samples: {len(train_indices)}"
            )
            start_time = time.time()
            best_val_dice, completed_epochs = train_model(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                patience=args.patience,
                labeled_indices=labeled_indices,
                pseudo_indices=pseudo_indices,
                device=device,
            )
            test_dice, test_iou = evaluate(model, test_loader, device)
            elapsed = time.time() - start_time

            metrics_writer.writerow(
                {
                    "stage": stage,
                    "round": round_index,
                    "n_labeled": len(labeled_indices),
                    "n_pseudo": len(pseudo_indices),
                    "val_dice": f"{best_val_dice:.6f}",
                    "test_dice": f"{test_dice:.6f}",
                    "test_iou": f"{test_iou:.6f}",
                    "epochs": completed_epochs,
                    "seconds": f"{elapsed:.2f}",
                }
            )
            metrics_file.flush()
            print(
                f"[{stage}] best_val_dice={best_val_dice:.4f} | "
                f"test_dice={test_dice:.4f} | test_iou={test_iou:.4f} | "
                f"{elapsed:.1f}s"
            )

        initial_indices = select_largest_area(pool_indices, areas, args.init_size)
        labeled_indices = {int(index) for index in initial_indices}
        record_selection("init", 0, initial_indices)
        print(
            "Initial labeled indices (largest pseudo-mask areas): "
            f"{[int(index) for index in initial_indices]}"
        )
        run_stage("init", 0, labeled_indices, set(), args.epochs)

        for round_index in range(1, args.rounds + 1):
            candidates = np.asarray(
                sorted(set(pool_indices.tolist()) - labeled_indices), dtype=np.int64
            )
            if candidates.size == 0:
                print(f"\nAL round {round_index}: no unlabeled samples left; stopping.")
                break

            selected = select_largest_area(candidates, areas, args.query_size)
            if len(selected) < args.query_size:
                print(
                    f"\nAL round {round_index}: only {len(selected)} sample(s) left "
                    f"(requested {args.query_size})."
                )
            record_selection(f"al_round_{round_index}", round_index, selected)
            labeled_indices.update(int(index) for index in selected)
            print(
                f"\nAL round {round_index}: selected {len(selected)} sample(s), "
                f"{len(labeled_indices)} labeled in total."
            )
            run_stage(f"al_round_{round_index}", round_index, labeled_indices, set(), args.epochs)

        if args.use_ssl:
            remaining = np.asarray(
                sorted(set(pool_indices.tolist()) - labeled_indices), dtype=np.int64
            )
            if remaining.size > 0:
                pseudo_indices = {int(index) for index in remaining}
                ordered_remaining = select_largest_area(remaining, areas, len(remaining))
                record_selection("ssl", args.rounds + 1, ordered_remaining)
                run_stage(
                    "ssl",
                    args.rounds + 1,
                    labeled_indices,
                    pseudo_indices,
                    ssl_epochs,
                )
            else:
                print("\nSSL stage skipped: no unlabeled pool samples left.")

    print(f"\nDone. Metrics: {metrics_path}")
    print(f"Selected samples: {selections_path}")
    print(f"Arguments: {args_path}")


if __name__ == "__main__":
    main()
