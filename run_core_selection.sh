#!/usr/bin/env bash
set -euo pipefail

TOPK=15
QUERY=${QUERY:-10_10_s.bmp}
MAHAL_LAMBDA=${MAHAL_LAMBDA:-1e-6}
# mAP control: default on. Override via env EVAL_MAP or CLI flags --map/--no-map
EVAL_MAP=${EVAL_MAP:-1}
# Optional 1st arg: --map or --no-map
if [[ ${1-} == "--no-map" ]]; then EVAL_MAP=0; fi
if [[ ${1-} == "--map" ]]; then EVAL_MAP=1; fi

py() { python "$@"; }

vs_eval() {
  local sub="$1"; shift
  local metric="$1"; shift
  local extra=("$@")
  local eval_flag
  if [[ "${EVAL_MAP}" -eq 1 ]]; then eval_flag=--eval-map; else eval_flag=--eval; fi
  local cmd=(python Skeleton_Python/cvpr_visualsearch.py
    --desc-subfolder "${sub}"
    --topk "${TOPK}"
    --no-show
    --plot-query "${QUERY}"
    ${eval_flag}
    --vs-metric "${metric}"
    --eval-metric "${metric}")
  if (( ${#extra[@]} )); then cmd+=("${extra[@]}"); fi
  "${cmd[@]}"
}

echo "[Compute] Global hist rgb8/16/32"
for b in 8 16 32; do
  py Skeleton_Python/cvpr_computedescriptors.py global_color_hist --bins "$b"
done

echo "[Compute] Spatial 2x2 RGB colorOnly, grad8/16/32, HSV grad16, 3x3 RGB grad16, LBP r1/r2"
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 2x2 --bins 8 --colorspace rgb --texture none --color-only
for ob in 8 16 32; do
  py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 2x2 --bins 8 --colorspace rgb --texture grad --orient-bins "$ob"
done
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 2x2 --bins 8 --colorspace hsv --texture grad --orient-bins 16
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 3x3 --bins 8 --colorspace rgb --texture grad --orient-bins 16
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 2x2 --bins 8 --colorspace rgb --texture lbp --lbp-points 8 --lbp-radius 1
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 2x2 --bins 8 --colorspace rgb --texture lbp --lbp-points 8 --lbp-radius 2

echo "[PCA] Spatial 2x2 RGB/HSV grad16 -> 128D whiten"
py Skeleton_Python/pca.py --desc-subfolder spatialGrid/2x2_rgb_c8_grad16u --dim 128 --whiten
py Skeleton_Python/pca.py --desc-subfolder spatialGrid/2x2_hsv_c8_grad16u --dim 128 --whiten

echo "[Compute] BoVW SIFT k200/k400/k800"
for k in 200 400 800; do
  py Skeleton_Python/cvpr_computedescriptors.py bow --codebook-size "$k" --detector sift --build-vocab --vocab-file "descriptor/BoVW/vocab_k${k}_sift.npz"
  py Skeleton_Python/cvpr_computedescriptors.py bow --codebook-size "$k" --detector sift --vocab-file "descriptor/BoVW/vocab_k${k}_sift.npz"
done

echo "[PCA] BoVW k400 -> 128D"
py Skeleton_Python/pca.py --desc-subfolder BoVW/sift_k400 --dim 128 --whiten

RUN_ID=${RUN_ID:-core-$(date +%m%d%H%M%S)-$$}
echo "[Eval] Hist Chi2+L2 [RUN_ID=${RUN_ID}]"
for b in 8 16 32; do for m in chi2 l2; do vs_eval "globalRGBhisto/rgb$b" "$m"; sleep 1; done; done

echo "[Eval] Spatial raw Chi2+L2 (selected)"
for sub in \
  spatialGrid/2x2_rgb_c8_colorOnly \
  spatialGrid/2x2_rgb_c8_grad8u \
  spatialGrid/2x2_rgb_c8_grad16u \
  spatialGrid/2x2_hsv_c8_grad16u \
  spatialGrid/3x3_rgb_c8_grad16u \
  spatialGrid/2x2_rgb_c8_lbp8r2; do
  for m in chi2 l2; do vs_eval "$sub" "$m"; sleep 0.3; done
done

echo "[Eval] Spatial PCA L2+Mahalanobis"
for sub in spatialGrid/2x2_rgb_c8_grad16u_pca128 spatialGrid/2x2_hsv_c8_grad16u_pca128; do
  vs_eval "$sub" l2; sleep 0.3
  vs_eval "$sub" mahal --mahal-lambda "$MAHAL_LAMBDA"; sleep 0.3
done

echo "[Eval] BoVW raw"
for k in 200 800; do vs_eval "BoVW/sift_k$k" chi2; sleep 0.3; done
for m in chi2 l2; do vs_eval "BoVW/sift_k400" "$m"; sleep 0.3; done

echo "[Eval] BoVW PCA"
for m in l2 mahal; do
  if [[ "$m" == mahal ]]; then vs_eval "BoVW/sift_k400_pca128" "$m" --mahal-lambda "$MAHAL_LAMBDA"; else vs_eval "BoVW/sift_k400_pca128" "$m"; fi; sleep 0.3
done

echo "[Aggregate]"
python tools/aggregate_evals.py --eval-root results/eval --out results/eval/REPORT_core_summary.csv
python tools/aggregate_evals.py --eval-root results/eval --out results/eval/REPORT_all_summary.csv
echo "Core selection done."
