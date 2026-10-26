# Minimal reproduction: `area_seg` active learning for segmentation

This repository is a small, self-contained reproduction of the **`area_seg`**
active learning strategy used in our segmentation experiments. Only U-Net is
kept; other models, strategies, and auxiliary losses were removed.

## Method

Each image in the training pool has a **pre-computed pseudo mask** (in the
original experiments these were produced by MedSAM and stored under
`P_masks_path`). The pseudo-mask area is

```text
area(i) = number of pixels with pseudo_mask(i) > 0.5
```

computed after resizing the pseudo mask to the training resolution with
nearest-neighbour interpolation.

The active learning procedure is:

1. **Initial labeled set** — select the `--init-size` pool images with the
   largest pseudo-mask areas, and use their real ground-truth masks.
2. **Train** U-Net on the currently labeled images with a per-sample Dice loss.
3. **Query** — among the remaining pool images, select the `--query-size`
   images with the largest pseudo-mask areas, then obtain their real masks.
4. **Repeat** step 2–3 for `--rounds` rounds. The model is reused (not
   re-initialized) across rounds, as in the original code.
5. **Optional SSL (`--use-ssl`)** — after the AL rounds, train once more using
   the pre-computed pseudo masks as labels for every remaining pool image. This
   corresponds to `SSL=True, SSL_Large_model=True` in the original experiment.

Because the score depends only on the pre-computed pseudo masks, `area_seg` is
a static ranking: each round reveals the next largest-area images. Retraining
between rounds still matters for the accuracy-vs-budget curve.

## Repository layout

```text
.
├── main.py          # argument parsing and the active learning loop
├── data.py          # paired data loading, splits, preprocessing
├── model.py         # U-Net from the original project (logits output)
├── area_seg.py      # pseudo-mask area computation and top-k selection
├── train.py         # Dice loss, training, validation, Dice/IoU metrics
├── requirements.txt
└── README.md
```

## Data format

```text
data_root/
├── images/          # e.g. case_001.jpg
├── masks/           # e.g. case_001.png   ground-truth binary masks
└── pseudo_masks/    # e.g. case_001.png   pre-computed binary pseudo masks
```

- Files are paired by **stem** (file name without extension), so the
  extensions may differ between the three directories.
- Masks should be binary. Values greater than 127 are treated as foreground
  after nearest-neighbour resizing.
- The code does **not** generate pseudo masks. Prepare them beforehand, e.g.
  with the same MedSAM pipeline used in the original experiments, and place
  them in `pseudo_masks/`.
- The AL pool / validation / test split is deterministic given `--seed`,
  `--val-ratio`, and `--test-ratio`. If you need the official benchmark
  splits, adapt `data.py` or pre-split the files.

## Installation

Python 3.8+ is recommended.

```bash
pip install -r requirements.txt
```

## Run

Original ISIC2017/ISIC2018-style settings:

```bash
python main.py \
  --data-root /path/to/data \
  --img-size 224 \
  --init-size 15 \
  --query-size 15 \
  --rounds 4 \
  --epochs 100 \
  --lr 0.01 \
  --batch-size 4 \
  --seed 42 \
  --use-ssl
```

Quick smoke run (small images, few epochs, CPU):

```bash
python main.py \
  --data-root /path/to/data \
  --img-size 64 \
  --init-size 4 \
  --query-size 4 \
  --rounds 2 \
  --epochs 2 \
  --patience 0 \
  --batch-size 2 \
  --device cpu
```

`--patience 0` disables early stopping. All options are listed by
`python main.py --help`.

## Outputs

Results are written to `outputs/seed<SEED>/`:

- `metrics.csv` — one row per stage (initial, each AL round, optional SSL)
  with the number of real/pseudo labels, validation Dice, test Dice/IoU,
  completed epochs, and runtime.
- `selected_samples.csv` — the images selected at every stage, in selection
  order, with their pseudo-mask areas.
- `args.json` — the exact arguments of the run.

## Differences from the full experimental code

- Only U-Net (`Unet(3, 1)`) is included.
- Only the `area_seg` strategy is included; all other query strategies,
  Siamese/LPIPS modules, and large-model SSL variants were removed.
- Augmentation is reduced to random horizontal/vertical flips. The full code
  also used rotate90, color jitter, brightness/contrast, and Gaussian blur.
- Metrics are limited to Dice and IoU (the full code also reported
  Hausdorff95, sensitivity, specificity, and F1).
- No model checkpointing or per-image score CSV export.
- The training/validation/test split is ratio-based rather than tied to a
  particular dataset directory layout.
- The AL loop always runs `--rounds` rounds. The original also stopped early
  after 20 rounds without validation improvement.
- The SSL stage uses `--ssl-epochs`, which defaults to `--epochs`; the original
  used `epochs_SSL=100` for the large-model SSL setting.

## Reproducibility notes

- Python, NumPy, PyTorch, and CUDA are seeded; cuDNN deterministic mode is
  enabled.
- DataLoader shuffling uses a per-stage seeded generator.
- `--num-workers 0` is the default, avoiding worker-level RNG differences.
- Exact GPU results can still vary across hardware and library versions.
