#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Bulk pipeline for computing descriptors, optional PCA, and VS+Eval.
# Usage: bash run_all.sh

# -------- settings you can tweak --------
TOPK=${TOPK:-15}
QUERY=${QUERY:-10_10_s.bmp}
DO_PCA=${DO_PCA:-1}              # 1=also compute PCA versions
PCA_DIM=${PCA_DIM:-128}
PCA_WHITEN=${PCA_WHITEN:-1}      # 1=whiten
MAHAL_LAMBDA=${MAHAL_LAMBDA:-1e-6}
EVAL_MAP=${EVAL_MAP:-0}            # 1=compute mAP during evaluation

# what to compute (space-separated)
HIST_BINS=(${HIST_BINS:-4 8 16 32 64})
GRID_LIST=(${GRID_LIST:-2x2})
GRID_COLSP=(${GRID_COLSP:-rgb hsv})
GRID_TEXTURES=(${GRID_TEXTURES:-grad})    # options: grad, lbp
GRAD_BINS=(${GRAD_BINS:-8 16 32})
LBP_POINTS=(${LBP_POINTS:-8})
LBP_RADIUS=(${LBP_RADIUS:-1 2})

# BoVW configs
BOW_DETECTORS=(${BOW_DETECTORS:-sift harris})
BOW_SIZES=(${BOW_SIZES:-200 400 600 800})

# evaluation metrics
METRICS_HIST=(${METRICS_HIST:-l2 chi2})
METRICS_BOVW=(${METRICS_BOVW:-l2 chi2})
METRICS_PCA=(${METRICS_PCA:-l2 mahal})
# ----------------------------------------

py() { python "$@"; }

vs_eval() {
  local sub="$1"; shift
  local metric="$1"; shift
  local extra=("$@")
  echo "[VS+EVAL] ${sub} metric=${metric}"
  py Skeleton_Python/cvpr_visualsearch.py \
    --desc-subfolder "${sub}" \
    --topk "${TOPK}" \
    --no-show \
    --plot-query "${QUERY}" \
    $( [[ "${EVAL_MAP}" -eq 1 ]] && echo "--eval-map" || echo "--eval" ) \
    --vs-metric "${metric}" \
    --eval-metric "${metric}" \
    "${extra[@]}"
}

ensure_pca() {
  local sub="$1"
  local out="descriptor/${sub}_pca${PCA_DIM}"
  if [[ "${DO_PCA}" -eq 1 ]]; then
    if [[ ! -d "${out}" ]]; then
      echo "[PCA] ${sub} -> ${out}"
      if [[ "${PCA_WHITEN}" -eq 1 ]]; then
        py Skeleton_Python/pca.py --desc-subfolder "${sub}" --dim "${PCA_DIM}" --whiten
      else
        py Skeleton_Python/pca.py --desc-subfolder "${sub}" --dim "${PCA_DIM}"
      fi
    else
      echo "[PCA] Skip (exists) ${out}"
    fi
  fi
}

# 1) Global colour histogram
for b in "${HIST_BINS[@]}"; do
  sub="globalRGBhisto/rgb${b}"
  if [[ ! -d "descriptor/${sub}" ]]; then
    echo "[COMPUTE] ${sub}"
    py Skeleton_Python/cvpr_computedescriptors.py global_color_hist --bins "${b}"
  else
    echo "[SKIP] ${sub}"
  fi

  ensure_pca "${sub}"

  for m in "${METRICS_HIST[@]}"; do
    vs_eval "${sub}" "${m}"
  done
  if [[ "${DO_PCA}" -eq 1 ]]; then
    for m in "${METRICS_PCA[@]}"; do
      if [[ "${m}" == "mahal" ]]; then
        vs_eval "${sub}_pca${PCA_DIM}" "${m}" --mahal-lambda "${MAHAL_LAMBDA}"
      else
        vs_eval "${sub}_pca${PCA_DIM}" "${m}"
      fi
    done
  fi
done

# 2) Spatial Grid
for grid in "${GRID_LIST[@]}"; do
  for cs in "${GRID_COLSP[@]}"; do

    # Gradient texture variants
    if printf '%s\n' "${GRID_TEXTURES[@]}" | grep -q '^grad$'; then
      for ob in "${GRAD_BINS[@]}"; do
        sub="spatialGrid/${grid}_${cs}_c8_grad${ob}u"
        if [[ ! -d "descriptor/${sub}" ]]; then
          echo "[COMPUTE] ${sub}"
          py Skeleton_Python/cvpr_computedescriptors.py spatial_grid \
            --grid "${grid}" --bins 8 --colorspace "${cs}" \
            --texture grad --orient-bins "${ob}"
        else
          echo "[SKIP] ${sub}"
        fi

        ensure_pca "${sub}"
        for m in "${METRICS_HIST[@]}"; do vs_eval "${sub}" "${m}"; done
        if [[ "${DO_PCA}" -eq 1 ]]; then
          for m in "${METRICS_PCA[@]}"; do
            if [[ "${m}" == "mahal" ]]; then
              vs_eval "${sub}_pca${PCA_DIM}" "${m}" --mahal-lambda "${MAHAL_LAMBDA}"
            else
              vs_eval "${sub}_pca${PCA_DIM}" "${m}"
            fi
          done
        fi
      done
    fi

    # LBP texture variants
    if printf '%s\n' "${GRID_TEXTURES[@]}" | grep -q '^lbp$'; then
      for p in "${LBP_POINTS[@]}"; do
        for r in "${LBP_RADIUS[@]}"; do
          sub="spatialGrid/${grid}_${cs}_c8_lbp${p}r${r}"
          if [[ ! -d "descriptor/${sub}" ]]; then
            echo "[COMPUTE] ${sub}"
            py Skeleton_Python/cvpr_computedescriptors.py spatial_grid \
              --grid "${grid}" --bins 8 --colorspace "${cs}" \
              --texture lbp --lbp-points "${p}" --lbp-radius "${r}"
          else
            echo "[SKIP] ${sub}"
          fi

          ensure_pca "${sub}"
          for m in "${METRICS_HIST[@]}"; do vs_eval "${sub}" "${m}"; done
          if [[ "${DO_PCA}" -eq 1 ]]; then
            for m in "${METRICS_PCA[@]}"; do
              if [[ "${m}" == "mahal" ]]; then
                vs_eval "${sub}_pca${PCA_DIM}" "${m}" --mahal-lambda "${MAHAL_LAMBDA}"
              else
                vs_eval "${sub}_pca${PCA_DIM}" "${m}"
              fi
            done
          fi
        done
      done
    fi

  done
done

# 3) BoVW: build vocab + compute, then PCA and evaluate
for det in "${BOW_DETECTORS[@]}"; do
  for k in "${BOW_SIZES[@]}"; do
    vocab="descriptor/BoVW/vocab_k${k}_${det}.npz"
    sub="BoVW/${det}_k${k}"

    if [[ ! -f "${vocab}" ]]; then
      echo "[VOCAB] ${det} K=${k}"
      py Skeleton_Python/cvpr_computedescriptors.py bow \
        --codebook-size "${k}" --detector "${det}" \
        --build-vocab --vocab-file "${vocab}"
    else
      echo "[SKIP] vocab ${vocab}"
    fi

    if [[ ! -d "descriptor/${sub}" ]]; then
      echo "[COMPUTE] ${sub}"
      py Skeleton_Python/cvpr_computedescriptors.py bow \
        --codebook-size "${k}" --detector "${det}" \
        --vocab-file "${vocab}"
    else
      echo "[SKIP] ${sub}"
    fi

    ensure_pca "${sub}"
    for m in "${METRICS_BOVW[@]}"; do vs_eval "${sub}" "${m}"; done
    if [[ "${DO_PCA}" -eq 1 ]]; then
      for m in "${METRICS_PCA[@]}"; do
        if [[ "${m}" == "mahal" ]]; then
          vs_eval "${sub}_pca${PCA_DIM}" "${m}" --mahal-lambda "${MAHAL_LAMBDA}"
        else
          vs_eval "${sub}_pca${PCA_DIM}" "${m}"
        fi
      done
    fi

  done
done

echo "All done."
