#!/usr/bin/env bash
set -euo pipefail

TOPK=15
QUERY=${QUERY:-10_10_s.bmp}
MAHAL_LAMBDA=${MAHAL_LAMBDA:-1e-6}
EVAL_MAP=${EVAL_MAP:-1}  # --map/--no-map supported
if [[ ${1-} == "--no-map" ]]; then EVAL_MAP=0; fi
if [[ ${1-} == "--map" ]]; then EVAL_MAP=1; fi

vs_eval() {
  local sub="$1"; shift
  local metric="$1"; shift
  local extra=("$@")
  local eval_flag; if [[ "${EVAL_MAP}" -eq 1 ]]; then eval_flag=--eval-map; else eval_flag=--eval; fi
  local cmd=(python Skeleton_Python/cvpr_visualsearch.py
    --desc-subfolder "${sub}"
    --topk "${TOPK}"
    --no-show
    --plot-query "${QUERY}"
    ${eval_flag}
    --vs-metric "${metric}"
    --eval-metric "${metric}")
  if (( ${#extra[@]} )); then cmd+=("${extra[@]}"); fi
  "${cmd[@]}" --run-id "${RUN_ID}"
  sleep 1
}

RUN_ID=${RUN_ID:-core-eval-$(date +%m%d%H%M%S)-$$}
echo "[Eval-only] Global hist rgb8/16/32 (Chi2+L2) [RUN_ID=${RUN_ID}]"
for b in 8 16 32; do
  for m in chi2 l2; do vs_eval "globalRGBhisto/rgb$b" "$m"; done
done

echo "[Eval-only] Spatial raw (selected) (Chi2+L2)"
for sub in \
  spatialGrid/2x2_rgb_c8_colorOnly \
  spatialGrid/2x2_rgb_c8_grad8u \
  spatialGrid/2x2_rgb_c8_grad16u \
  spatialGrid/2x2_hsv_c8_grad16u \
  spatialGrid/3x3_rgb_c8_grad16u \
  spatialGrid/2x2_rgb_c8_lbp8r2; do
  for m in chi2 l2; do vs_eval "$sub" "$m"; done
done

echo "[Eval-only] Spatial PCA (L2+Mahalanobis)"
for sub in spatialGrid/2x2_rgb_c8_grad16u_pca128 spatialGrid/2x2_hsv_c8_grad16u_pca128; do
  vs_eval "$sub" l2
  vs_eval "$sub" mahal --mahal-lambda "$MAHAL_LAMBDA"
done

echo "[Eval-only] BoVW raw"
for k in 200 800; do vs_eval "BoVW/sift_k$k" chi2; done
for m in chi2 l2; do vs_eval "BoVW/sift_k400" "$m"; done

echo "[Eval-only] BoVW PCA (L2+Mahalanobis)"
for m in l2 mahal; do
  if [[ "$m" == mahal ]]; then vs_eval "BoVW/sift_k400_pca128" "$m" --mahal-lambda "$MAHAL_LAMBDA"; else vs_eval "BoVW/sift_k400_pca128" "$m"; fi
done

python tools/aggregate_evals.py --eval-root results/eval --out results/eval/REPORT_core_evalonly_summary.csv
python tools/aggregate_evals.py --eval-root results/eval --out results/eval/REPORT_all_summary.csv
echo "Eval-only run complete."
