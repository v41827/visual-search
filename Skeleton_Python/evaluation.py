import os
import sys
import csv
import argparse
from datetime import datetime
from typing import List, Tuple, Optional
import re
import shlex

import numpy as np
import scipy.io as sio

# Ensure local imports work when running as a script
sys.path.append(os.path.dirname(__file__))
from cvpr_compare import cvpr_compare, l2_distance, l1_distance, chi_square_distance, histogram_intersection


IMAGE_FOLDER = os.path.join('MSRC_ObjCategImageDatabase_v2', 'Images')
DESCRIPTOR_FOLDER = 'descriptor'


def _load_descriptors(desc_dir: str) -> Tuple[np.ndarray, List[str]]:
    ALLFEAT = []
    ALLFILES: List[str] = []
    for filename in sorted(os.listdir(desc_dir)):
        if not filename.endswith('.mat'):
            continue
        mat_path = os.path.join(desc_dir, filename)
        try:
            mat = sio.loadmat(mat_path)
        except Exception:
            continue
        F = mat.get('F')
        if F is None:
            continue
        F = np.asarray(F)
        if F.ndim > 1:
            F = F.reshape(-1)
        ALLFEAT.append(F.astype(np.float64))

        base = mat.get('file')
        if isinstance(base, np.ndarray):
            if base.dtype.kind in ('U', 'S', 'O'):
                try:
                    base = str(base.squeeze())
                except Exception:
                    base = None
            else:
                try:
                    base = ''.join(chr(int(c)) for c in base.ravel())
                except Exception:
                    base = None
        if not base:
            base = os.path.splitext(filename)[0] + '.bmp'
        ALLFILES.append(str(base))
    return np.vstack(ALLFEAT), ALLFILES


def _labels_from_filenames(files: List[str]) -> np.ndarray:
    labels = []
    for f in files:
        bn = os.path.basename(f)
        token = bn.split('_')[0]
        try:
            labels.append(int(token))
        except Exception:
            labels.append(-1)
    return np.array(labels, dtype=int)


def _pairwise_l2(A: np.ndarray) -> np.ndarray:
    # Compute pairwise L2 distance matrix for rows of A
    # Returns NxN matrix with zeros on diagonal
    a2 = np.sum(A * A, axis=1, keepdims=True)  # (N,1)
    d2 = a2 + a2.T - 2.0 * (A @ A.T)
    d2[d2 < 0] = 0
    D = np.sqrt(d2, dtype=np.float64)
    return D


def evaluate(
    feats: np.ndarray,
    files: List[str],
    topk: int = 15,
    metric: str = 'l2',
    compute_map: bool = False,
    mahal_lambda: float = 1e-6,
    plot_query: Optional[str] = None,
) -> dict:
    N = feats.shape[0]
    labels = _labels_from_filenames(files)
    classes = np.unique(labels[labels >= 0])
    C = int(classes.max()) if classes.size else 0

    # Distance matrix
    if metric == 'l2':
        D = _pairwise_l2(feats)
    elif metric == 'mahal':
        # Compute Mahalanobis via whitening transform
        X = feats.astype(np.float64)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        # Covariance and regularization
        C = np.cov(Xc, rowvar=False)
        # Small regularizer for stability
        lam = float(mahal_lambda)
        # eigh for symmetric covariance
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 0.0, None)
        denom = np.sqrt(vals + lam)
        # Whitening matrix W = U * diag(1/sqrt(vals+lam))
        W = (vecs / denom)
        # Y = Xc @ U @ diag(1/sqrt(vals+lam)) = Xc @ W
        Y = Xc @ W
        D = _pairwise_l2(Y)
    else:
        # Fallback to scalar comparator for each pair
        D = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i, N):
                if i == j:
                    continue
                if metric == 'l1':
                    d = l1_distance(feats[i], feats[j])
                elif metric == 'chi2':
                    d = chi_square_distance(feats[i], feats[j])
                elif metric == 'histint':
                    d = histogram_intersection(feats[i], feats[j])
                else:
                    d = cvpr_compare(feats[i], feats[j])
                D[i, j] = D[j, i] = d

    # Metrics accumulators
    pr_points = np.zeros((topk, 2), dtype=np.float64)  # precision, recall averaged across queries
    top1_correct = 0
    top5_correct = 0
    cm_labels = np.unique(labels)
    cm_labels = cm_labels[cm_labels >= 0]
    label_to_idx = {lab: idx for idx, lab in enumerate(sorted(cm_labels))}
    cm = np.zeros((len(label_to_idx), len(label_to_idx)), dtype=np.int64)
    P_at_15 = []
    R_at_15 = []
    AP_list = []

    # Normalize plot_query name if any
    plot_query_norm = os.path.basename(plot_query).lower() if plot_query else None
    per_query_curve = None  # will hold dict for selected query

    for q in range(N):
        # sort by distance, exclude self
        order = np.argsort(D[q])
        order = order[order != q]
        qlab = labels[q]
        top = order[:topk]
        top_labels = labels[top]

        # Confusion (top-1)
        if top.size > 0 and qlab >= 0 and top_labels[0] >= 0:
            cm[label_to_idx.get(qlab, 0), label_to_idx.get(top_labels[0], 0)] += 1

        # Top-k accuracy
        if top.size > 0:
            top1_correct += int(top_labels[0] == qlab)
            top5_correct += int(np.any(top_labels[: min(5, top.size)] == qlab))

        # Relevant totals (same class excluding self)
        rel_total = int(np.sum(labels == qlab)) - 1
        rel_total = max(rel_total, 0)

        # PR at each rank up to topk
        rel_cum = 0
        q_pr = np.zeros((topk, 2), dtype=np.float64)
        for r in range(min(topk, top.size)):
            rel_cum += int(top_labels[r] == qlab)
            prec = rel_cum / float(r + 1)
            rec = (rel_cum / float(rel_total)) if rel_total > 0 else 0.0
            pr_points[r, 0] += prec
            pr_points[r, 1] += rec
            q_pr[r, 0] = prec
            q_pr[r, 1] = rec

        # P@15, R@15
        k15 = min(15, top.size)
        rel15 = int(np.sum(top_labels[:k15] == qlab))
        P_at_15.append(rel15 / float(k15) if k15 > 0 else 0.0)
        R_at_15.append(rel15 / float(rel_total) if rel_total > 0 else 0.0)

        # AP (full ranking)
        if compute_map and rel_total > 0:
            hits = 0
            sum_prec = 0.0
            for rank, j in enumerate(order, start=1):
                if labels[j] == qlab:
                    hits += 1
                    sum_prec += hits / rank
            AP_list.append(sum_prec / rel_total if rel_total > 0 else 0.0)

        # Capture per-query curve for requested filename
        if plot_query_norm is not None and os.path.basename(files[q]).lower() == plot_query_norm:
            per_query_curve = {
                'filename': files[q],
                'pr_points': q_pr,
                'P_at_15': P_at_15[-1] if P_at_15 else 0.0,
                'R_at_15': R_at_15[-1] if R_at_15 else 0.0,
            }
            if compute_map and rel_total > 0:
                # recompute AP for this query to store explicitly
                hits = 0
                sum_prec = 0.0
                for rank, j in enumerate(order, start=1):
                    if labels[j] == qlab:
                        hits += 1
                        sum_prec += hits / rank
                per_query_curve['AP'] = sum_prec / rel_total if rel_total > 0 else 0.0

    # Average PR points across queries
    pr_points /= float(N)

    results = {
        'pr_points': pr_points,  # shape (topk, 2)
        'top1_acc': top1_correct / float(N),
        'top5_acc': top5_correct / float(N),
        'mean_P_at_15': float(np.mean(P_at_15)) if P_at_15 else 0.0,
        'mean_R_at_15': float(np.mean(R_at_15)) if R_at_15 else 0.0,
        'mAP': float(np.mean(AP_list)) if AP_list else None,
        'confusion_matrix': cm,
        'confusion_labels': sorted(label_to_idx.keys()),
        'N': int(N),
        'per_query_curve': per_query_curve,
    }
    return results


def _descriptor_label(desc_subfolder: str) -> str:
    s = desc_subfolder.replace('\\', '/').strip('/')
    low = s.lower()
    m = re.match(r'^globalrgbhisto/rgb(\d+)$', low)
    if m:
        return f"hist_rgb_{m.group(1)}bins"
    if low.startswith('spatialgrid/'):
        return 'spatial_grid_' + low.split('/', 1)[1].replace('/', '_')
    if low.startswith('bovw/'):
        return 'bow_' + low.split('/', 1)[1].replace('/', '_')
    return low.replace('/', '_')


def _save_eval(results_dir: str, desc_subfolder: str, res: dict, metric: str = 'l2', topk: int = 15, run_command: Optional[str] = None, run_id: Optional[str] = None):
    now = datetime.now()
    ts = now.strftime('%m%d-%H%M')
    ts_full = now.strftime('%m%d%H%M%S')
    # Use seconds-resolution folder to avoid collisions when running multiple
    # evaluations of the same descriptor within the same minute (different metrics).
    out_dir = os.path.join(results_dir, 'eval', desc_subfolder, ts_full)
    os.makedirs(out_dir, exist_ok=True)

    # Save command used to run
    try:
        if run_command is None:
            script_rel = os.path.relpath(__file__, start=os.getcwd())
            run_command = "python " + script_rel + " " + " ".join(shlex.quote(a) for a in sys.argv[1:])
        with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
            f.write(run_command + "\n")
    except Exception:
        pass

    # PR points
    pr_points = res['pr_points']
    label = _descriptor_label(desc_subfolder)
    with open(os.path.join(out_dir, 'pr_points.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['rank', 'precision', 'recall'])
        for r in range(pr_points.shape[0]):
            w.writerow([r + 1, float(pr_points[r, 0]), float(pr_points[r, 1])])

    # Optional: per-query PR curve
    pq = res.get('per_query_curve')
    if pq is not None:
        qname = os.path.basename(str(pq.get('filename', 'query')))
        qstem = os.path.splitext(qname)[0]
        qpoints = pq.get('pr_points')
        if isinstance(qpoints, np.ndarray):
            with open(os.path.join(out_dir, f'pr_points_{qstem}.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['rank', 'precision', 'recall'])
                for r in range(qpoints.shape[0]):
                    w.writerow([r + 1, float(qpoints[r, 0]), float(qpoints[r, 1])])

    # Summary
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['metric', 'value'])
        # config header rows for traceability
        w.writerow(['desc_subfolder', desc_subfolder])
        w.writerow(['descriptor_label', _descriptor_label(desc_subfolder)])
        w.writerow(['metric_name', metric.upper()])
        w.writerow(['topk', topk])
        w.writerow(['N', res.get('N', 0)])
        w.writerow(['timestamp', ts])
        w.writerow(['timestamp_full', ts_full])
        if run_id:
            w.writerow(['run_id', run_id])
        # summary metrics
        w.writerow(['top1_acc', res['top1_acc']])
        w.writerow(['top5_acc', res['top5_acc']])
        # mean_P_at_15 is mean precision@15 over queries (not mAP)
        w.writerow(['mean_P_at_15', res['mean_P_at_15']])
        w.writerow(['mean_R_at_15', res['mean_R_at_15']])
        if res['mAP'] is not None:
            w.writerow(['mAP', res['mAP']])

    # Confusion matrix
    cm = res['confusion_matrix']
    labels = res['confusion_labels']
    with open(os.path.join(out_dir, 'confusion_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow([''] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + list(map(int, cm[i].tolist())))

    # Optional plots
    try:
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except Exception:
            sns = None

        # PR curve
        plt.figure()
        plt.plot(pr_points[:, 1], pr_points[:, 0], marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.title(f"Average PR Curve — {label}")
        plt.grid(True, ls='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        # Add config info below plot
        subtitle = (
            f"metric={metric.upper()}, K={topk}, N={res.get('N', 0)}, "
            f"Top-1={res['top1_acc']*100:.1f}%, Top-5={res['top5_acc']*100:.1f}%"
        )
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=9)
        plt.savefig(os.path.join(out_dir, 'pr_curve.png'))
        plt.close()

        # Per-query PR curve plot if provided
        if pq is not None and isinstance(qpoints, np.ndarray):
            plt.figure()
            plt.plot(qpoints[:, 1], qpoints[:, 0], marker='o')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.title(f"PR Curve — {qname} — {label}")
            plt.grid(True, ls='--', alpha=0.4)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=9)
            plt.savefig(os.path.join(out_dir, f'pr_curve_{qstem}.png'))
            plt.close()

        # label and subtitle defined above for reuse

        # Confusion matrix (matplotlib version)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f"Confusion Matrix (Top-1) — {label}")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        # Annotate numbers on each cell
        try:
            thresh = cm.max() / 2.0 if cm.size else 0.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = int(cm[i, j])
                    plt.text(
                        j,
                        i,
                        str(val),
                        ha='center',
                        va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=8,
                    )
        except Exception:
            pass
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=9)
        plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
        plt.close()

        # Confusion matrix (seaborn heatmap, if available)
        if sns is not None:
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='YlGnBu',
                cbar=True,
                xticklabels=labels,
                yticklabels=labels,
                linewidths=0.5,
                linecolor='white',
                square=True,
            )
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            ax.set_title(f"Confusion Matrix (Top-1) — {label}")
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=9)
            plt.savefig(os.path.join(out_dir, 'confusion_matrix_heatmap.png'))
            plt.close()
    except Exception:
        pass

    print(f"Saved evaluation to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval: PR@K, PR curve, CM, Top1/Top5, optional mAP')
    parser.add_argument('--desc-subfolder', required=True, help='Subfolder inside descriptor/, e.g., globalRGBhisto/rgb8')
    parser.add_argument('--topk', type=int, default=15, help='Evaluate metrics up to this rank')
    parser.add_argument('--results-dir', default='results', help='Base folder to save evaluation results')
    parser.add_argument('--metric', choices=['l2', 'l1', 'chi2', 'histint', 'mahal'], default='l2')
    parser.add_argument('--compute-map', action='store_true', help='Also compute mAP (slower)')
    parser.add_argument('--limit-queries', type=int, default=None, help='Optional: only evaluate first N queries')
    parser.add_argument('--plot-query', default='10_10_s.bmp', help='Also save PR curve for this specific filename (default: 10_10_s.bmp)')
    parser.add_argument('--mahal-lambda', type=float, default=1e-6, help='Regularization lambda for Mahalanobis (adds lambda*I to covariance)')

    args = parser.parse_args()

    desc_dir = os.path.join(DESCRIPTOR_FOLDER, args.desc_subfolder)
    feats, files = _load_descriptors(desc_dir)
    if args.limit_queries is not None:
        feats = feats[: args.limit_queries]
        files = files[: args.limit_queries]

    # Pack lambda in a small trick: when metric is mahal, temporarily prepend it into sys.argv subtitle via _save_eval call
    res = evaluate(
        feats,
        files,
        topk=args.topk,
        metric=args.metric,
        compute_map=args.compute_map,
        mahal_lambda=args.mahal_lambda,
        plot_query=args.plot_query,
    )
    _save_eval(args.results_dir, args.desc_subfolder, res, metric=args.metric, topk=args.topk)


if __name__ == '__main__':
    main()
