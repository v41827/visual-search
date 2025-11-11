#!/usr/bin/env python3
"""
Batch utilities to (1) replot PR curves with consistent scaling/titles, and
(2) run searches + evals for chosen queries across all available descriptors.

Replotting reads existing pr_points.csv and summary.csv from results/eval/**/
and writes pr_curve_fixed.png (and pr_curve_<q>_fixed.png for per-query).

Search/eval runs use Skeleton_Python/cvpr_visualsearch.py with provided queries
and metrics, tagging outputs with a run-id to avoid collisions.

Examples
  # Replot all existing eval PR curves
  python tools/batch_eval_and_search.py --replot-existing

  # Replot + run searches for 5 queries across all descriptors, L2+Mahalanobis
  python tools/batch_eval_and_search.py --replot-existing --queries auto --metrics l2,mahal

  # Only run searches for specific queries
  python tools/batch_eval_and_search.py --queries 1_1_s.bmp,5_1_s.bmp,10_1_s.bmp,15_1_s.bmp,20_1_s.bmp --metrics l2
"""

import os
import csv
import argparse
from typing import List, Tuple

import numpy as np


def _read_summary(path: str) -> dict:
    vals = {}
    if not os.path.isfile(path):
        return vals
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            vals[row[0]] = row[1]
    return vals


def _read_pr_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # returns recall, precision arrays
    R, P = [], []
    with open(path, newline='') as f:
        it = csv.reader(f)
        header = next(it, None)
        for row in it:
            if len(row) < 3:
                continue
            try:
                # row: rank, precision, recall
                p = float(row[1]); r = float(row[2])
            except Exception:
                continue
            P.append(p); R.append(r)
    return np.array(R, dtype=float), np.array(P, dtype=float)


def replot_eval_folder(eval_dir: str) -> bool:
    import matplotlib.pyplot as plt
    pr_csv = os.path.join(eval_dir, 'pr_points.csv')
    summ = os.path.join(eval_dir, 'summary.csv')
    if not os.path.isfile(pr_csv) or not os.path.isfile(summ):
        return False
    meta = _read_summary(summ)
    label = meta.get('descriptor_label') or meta.get('desc_subfolder', '')
    metric = meta.get('metric_name', '')
    topk = meta.get('topk', '')
    N = meta.get('N', '')

    R, P = _read_pr_points(pr_csv)
    if R.size == 0:
        return False
    plt.figure()
    plt.plot(R, P, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(f'Average PR Curve — {label}')
    sub = f"metric={metric}, K={topk}, N={N}"
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.figtext(0.5, 0.01, sub, ha='center', fontsize=9)
    plt.savefig(os.path.join(eval_dir, 'pr_curve_fixed.png'))
    plt.close()

    # Per-query PR curves (if present)
    for fn in os.listdir(eval_dir):
        if not fn.startswith('pr_points_') or not fn.endswith('.csv'):
            continue
        qcsv = os.path.join(eval_dir, fn)
        qstem = fn[len('pr_points_'):-4]
        Rq, Pq = _read_pr_points(qcsv)
        if Rq.size == 0:
            continue
        plt.figure()
        plt.plot(Rq, Pq, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.title(f'PR Curve — {qstem} — {label}')
        plt.grid(True, ls='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.figtext(0.5, 0.01, sub, ha='center', fontsize=9)
        plt.savefig(os.path.join(eval_dir, f'pr_curve_{qstem}_fixed.png'))
        plt.close()
    return True


def discover_eval_folders(root: str = 'results/eval') -> List[str]:
    out = []
    if not os.path.isdir(root):
        return out
    for r, d, f in os.walk(root):
        if 'pr_points.csv' in f and 'summary.csv' in f:
            out.append(r)
    return out


def discover_descriptor_subfolders(desc_root: str = 'descriptor') -> List[str]:
    subs = set()
    if not os.path.isdir(desc_root):
        return []
    for r, d, f in os.walk(desc_root):
        if any(x.endswith('.mat') for x in f):
            # subfolder relative to desc_root
            subs.add(os.path.relpath(r, desc_root))
    return sorted(subs)


def choose_queries_auto(img_root: str, k: int = 5) -> List[str]:
    # Choose deterministically: first image from approximately evenly spaced classes
    imgs = sorted([fn for fn in os.listdir(img_root) if fn.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))])
    if not imgs:
        return []
    # group by class (first token number)
    from collections import defaultdict
    buckets = defaultdict(list)
    for fn in imgs:
        try:
            c = int(os.path.basename(fn).split('_')[0])
        except Exception:
            c = -1
        buckets[c].append(fn)
    classes = sorted([c for c in buckets.keys() if c >= 0])
    if not classes:
        return imgs[:k]
    picks = []
    for idx in np.linspace(0, len(classes)-1, num=k, dtype=int):
        c = classes[int(idx)]
        picks.append(buckets[c][0])
    return picks


def run_visual_search(desc_sub: str, query: str, metric: str, topk: int, run_id_suffix: str,
                      mahal_lambda: float = 1e-6):
    import subprocess
    cmd = [
        'python', os.path.join('Skeleton_Python', 'cvpr_visualsearch.py'),
        '--desc-subfolder', desc_sub,
        '--query', query,
        '--topk', str(topk),
        '--vs-metric', metric,
        '--eval-map',
        '--eval-metric', metric,
        '--plot-query', query,
        '--no-show',
        '--run-id', run_id_suffix,
    ]
    if metric == 'mahal':
        cmd += ['--mahal-lambda', str(mahal_lambda)]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description='Replot PR curves and run searches across descriptors')
    ap.add_argument('--replot-existing', action='store_true', help='Replot PR curves for all existing eval folders')
    ap.add_argument('--eval-root', default='results/eval')
    ap.add_argument('--desc-root', default='descriptor')
    ap.add_argument('--img-root', default=os.path.join('MSRC_ObjCategImageDatabase_v2', 'Images'))
    ap.add_argument('--queries', default='', help="Comma-separated list, or 'auto' to pick 5 deterministic queries")
    ap.add_argument('--metrics', default='l2,mahal')
    ap.add_argument('--topk', type=int, default=15)
    ap.add_argument('--mahal-lambda', type=float, default=1e-6)
    args = ap.parse_args()

    # 1) Replot existing
    if args.replot_existing:
        eval_dirs = discover_eval_folders(args.eval_root)
        cnt = 0
        for ed in eval_dirs:
            try:
                if replot_eval_folder(ed):
                    cnt += 1
            except Exception as e:
                print(f'[WARN] Failed to replot {ed}: {e}')
        print(f'Replotted PR curves in {cnt} eval folders')

    # 2) New searches/evals across descriptors
    if args.queries:
        if args.queries.strip().lower() == 'auto':
            queries = choose_queries_auto(args.img_root, k=5)
        else:
            queries = [q.strip() for q in args.queries.split(',') if q.strip()]
        if not queries:
            print('[INFO] No queries selected')
        else:
            metrics = [m.strip().lower() for m in args.metrics.split(',') if m.strip()]
            desc_subs = discover_descriptor_subfolders(args.desc_root)
            if not desc_subs:
                print('[INFO] No descriptor subfolders found under', args.desc_root)
            for desc in desc_subs:
                for q in queries:
                    qstem = os.path.splitext(os.path.basename(q))[0]
                    for m in metrics:
                        run_visual_search(desc, q, m, args.topk, run_id_suffix=f'batch_{qstem}_{m}', mahal_lambda=args.mahal_lambda)


if __name__ == '__main__':
    main()

