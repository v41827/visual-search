#!/usr/bin/env bash
set -euo pipefail

TOPK=15
QUERY=${QUERY:-10_10_s.bmp}
MAHAL_LAMBDA=${MAHAL_LAMBDA:-1e-6}
# mAP control: default on. Override via env EVAL_MAP or CLI flags --map/--no-map
EVAL_MAP=${EVAL_MAP:-1}
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

echo "[Compute] Optional descriptors"

# 1) Spatial 3x3 RGB grad16 -> PCA128
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 3x3 --bins 8 --colorspace rgb --texture grad --orient-bins 16
py Skeleton_Python/pca.py --desc-subfolder spatialGrid/3x3_rgb_c8_grad16u --dim 128 --whiten

# 2) Spatial 2x2 RGB colorOnly -> PCA128
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 2x2 --bins 8 --colorspace rgb --texture none --color-only
py Skeleton_Python/pca.py --desc-subfolder spatialGrid/2x2_rgb_c8_colorOnly --dim 128 --whiten

# 3) BoVW Harris k400 (+ PCA128)
py Skeleton_Python/cvpr_computedescriptors.py bow --codebook-size 400 --detector harris --build-vocab --vocab-file descriptor/BoVW/vocab_k400_harris.npz
py Skeleton_Python/cvpr_computedescriptors.py bow --codebook-size 400 --detector harris --vocab-file descriptor/BoVW/vocab_k400_harris.npz
py Skeleton_Python/pca.py --desc-subfolder BoVW/harris_k400 --dim 128 --whiten

# 4) Global hist rgb8 -> PCA128
py Skeleton_Python/cvpr_computedescriptors.py global_color_hist --bins 8
py Skeleton_Python/pca.py --desc-subfolder globalRGBhisto/rgb8 --dim 128 --whiten

# 5) Spatial 2x2 RGB grad16 signed (contrast)
py Skeleton_Python/cvpr_computedescriptors.py spatial_grid --grid 2x2 --bins 8 --colorspace rgb --texture grad --orient-bins 16 --signed

echo "[Eval] Optional set"

for sub in \
  spatialGrid/3x3_rgb_c8_grad16u_pca128 \
  spatialGrid/2x2_rgb_c8_colorOnly_pca128 \
  BoVW/harris_k400 \
  BoVW/harris_k400_pca128 \
  globalRGBhisto/rgb8_pca128 \
  spatialGrid/2x2_rgb_c8_grad16s; do
  case "$sub" in
    *pca128)
      vs_eval "$sub" l2; vs_eval "$sub" mahal --mahal-lambda "$MAHAL_LAMBDA" ;;
    BoVW/harris_k400)
      vs_eval "$sub" chi2; vs_eval "$sub" l2 ;;
    *)
      vs_eval "$sub" chi2; vs_eval "$sub" l2 ;;
  esac
done

python tools/aggregate_evals.py --eval-root results/eval --out results/eval/REPORT_optional_summary.csv
python tools/aggregate_evals.py --eval-root results/eval --out results/eval/REPORT_all_summary.csv
echo "Optional selection done."
