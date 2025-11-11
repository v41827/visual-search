# Visual Search on MSRC (EEE3032)

This repository contains a complete, reproducible pipeline for image retrieval on the MSRC Object Category dataset. It includes classic descriptors (global colour histograms, spatial grid with colour/texture, Bag of Visual Words), CNN‑based embeddings, a visual search demo, and evaluation utilities (PR curves, Top‑1/Top‑5, mAP, confusion matrices). A report aggregator produces a single CSV summarising all runs.

Quick Start
- Place images under `MSRC_ObjCategImageDatabase_v2/Images`.
- Compute a descriptor (choose one):
  - Global colour histogram: `python Skeleton_Python/cvpr_computedescriptors.py global_color_hist --bins 8 --dataset-folder MSRC_ObjCategImageDatabase_v2/Images --out-folder descriptor`
  - Spatial grid LBP 4×4: `python Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 4x4 --bins 8 --colorspace rgb --texture lbp --lbp-points 8 --lbp-radius 1 --dataset-folder MSRC_ObjCategImageDatabase_v2/Images --out-folder descriptor`
- Visual search (+eval, K=15):
  - `python Skeleton_Python/cvpr_visualsearch.py --desc-subfolder spatialGrid/4x4_rgb_c8_lbp8r1 --query 10_10_s.bmp --topk 15 --vs-metric chi2 --eval-map --eval-metric chi2 --plot-query 10_10_s.bmp --no-show`
- CNN embeddings (pretrained ResNet‑18 → 128‑D):
  - `python cnn_train.py --epochs 25 --batch-size 32`
- Aggregate report:
  - `bash refresh_report.sh`
- Replot PR curves and run multi‑query searches:
  - `python tools/batch_eval_and_search.py --replot-existing --queries auto --metrics l2,mahal`

Key folders and scripts
- `Skeleton_Python/`: core descriptors, visual search CLI, and evaluation code.
- `descriptor/`: per‑image descriptor files (`.mat`).
- `results/`: evaluation reports and visual search results (images, CSVs, plots).
- `cnn_train.py`: trains a pretrained ResNet‑18 to produce 128‑D descriptors and exports them to `descriptor/cnn/resnet18_embed128`.
- `cnn_scratch_train.py`: trains a simple 4‑layer CNN from scratch to produce 128‑D descriptors and exports them to `descriptor/cnn/conv_scratch_embed128`.
- `tools/batch_eval_and_search.py`: batch replot PR curves and run searches for multiple queries across all descriptors.
- `refresh_report.sh`: regenerates `results/eval/REPORT_all_summary.csv` by scanning all eval folders.

Prerequisites
- Python 3.9+ with the following packages: `numpy`, `scipy`, `opencv-python`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `torchvision`.
- Dataset placed under `MSRC_ObjCategImageDatabase_v2/Images` (591 images).
- On Apple Silicon, PyTorch MPS is automatically used when available.

Dataset Assumption
- Image labels are inferred from the filename: the integer before the first underscore is the class ID (1–20), e.g., `7_12_s.bmp → class 7`.

Running Descriptors
- Global colour histogram (RGB, 8 bins per channel):
  - `python Skeleton_Python/cvpr_computedescriptors.py global_color_hist --bins 8 --dataset-folder MSRC_ObjCategImageDatabase_v2/Images --out-folder descriptor`
- Spatial grid (e.g., 4×4, RGB, LBP 8r1):
  - `python Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 4x4 --bins 8 --colorspace rgb --texture lbp --lbp-points 8 --lbp-radius 1 --dataset-folder MSRC_ObjCategImageDatabase_v2/Images --out-folder descriptor`
- Bag of Visual Words (SIFT, K=200):
  - `python Skeleton_Python/cvpr_computedescriptors.py bow --codebook-size 200 --detector sift --dataset-folder MSRC_ObjCategImageDatabase_v2/Images --out-folder descriptor`

Visual Search (CLI)
- Use precomputed descriptors to rank all images for a given query and optionally evaluate:
  - `python Skeleton_Python/cvpr_visualsearch.py --desc-subfolder spatialGrid/4x4_rgb_c8_lbp8r1 --query 10_10_s.bmp --topk 15 --vs-metric chi2 --eval-map --eval-metric chi2 --plot-query 10_10_s.bmp --no-show`
- Outputs are written to `results/search/<desc>/<timestamp>_.../` and evaluation to `results/eval/<desc>/<timestamp>/` with PR curves, confusion matrices, and `summary.csv`.

Evaluation Only (Python API)
- The evaluation module computes PR@K, Top‑1/Top‑5, mAP, PR plots, and confusion matrices:
  - See `Skeleton_Python/evaluation.py`. CLI usage is via `cvpr_visualsearch.py --eval` or `--eval-map`.
- PR plots are standardised: axes are `[0,1]` and titles include descriptor/metric config.

CNN Descriptors
- Pretrained ResNet‑18 (128‑D):
  - Train + export + eval + search demo: `python cnn_train.py --epochs 25 --batch-size 32`
  - Exports `.mat` to `descriptor/cnn/resnet18_embed128/` and writes eval/search under `results/`.
  - Options: `--no-pretrained`, `--ckpt-out`, `--resume`, `--export-only`.
- Scratch CNN (Conv4, 128‑D):
  - Train CE only: `python cnn_scratch_train.py --epochs 80 --batch-size 32`
  - Optional Triplet fine‑tune: `--with-triplet --triplet-epochs 15`
  - Exports `.mat` to `descriptor/cnn/conv_scratch_embed128/`.

SVM Baseline (on descriptors)
- Train/test a linear SVM on any descriptor subfolder:
  - Spatial grid: `python Skeleton_Python/svm.py --desc-subfolder spatialGrid/4x4_rgb_c8_lbp8r1 --results-dir results --kernel linear --standardize --test-size 0.2`
  - CNN embeddings: `python Skeleton_Python/svm.py --desc-subfolder cnn/resnet18_embed128 --results-dir results --kernel linear --standardize --test-size 0.2`
- Outputs under `results/svm/<desc_label>/<timestamp>/` (per‑sample, summary, confusion matrices with descriptor label).

Batch Replot + Multi‑Query Search
- Replot PR curves for all existing evals and run new searches (5 auto queries) across all descriptors with L2 + Mahalanobis:
  - `python tools/batch_eval_and_search.py --replot-existing --queries auto --metrics l2,mahal`
- Custom query list and metrics (e.g., include chi2):
  - `python tools/batch_eval_and_search.py --queries 1_1_s.bmp,5_3_s.bmp,10_7_s.bmp,15_2_s.bmp,20_4_s.bmp --metrics l2,chi2,mahal`

Aggregate Report
- Regenerate a single CSV summarising all evaluation runs:
  - `bash refresh_report.sh`
- Output: `results/eval/REPORT_all_summary.csv` with metric‑first columns (distance, top1_acc, …) followed by config fields.

Notes
- Distance choices: histogram descriptors usually prefer `chi2`; deep embeddings prefer `l2` or `mahal`.
- Mahalanobis in this repo uses whitening with small regularisation (good for 128‑D CNN embeddings).
- All outputs are timestamped; nothing is overwritten. Use `--run-id` in `cvpr_visualsearch.py` for extra labels.
