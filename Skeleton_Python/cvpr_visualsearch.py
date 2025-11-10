import os
import sys
import csv
import argparse
from datetime import datetime
import numpy as np
import scipy.io as sio
import cv2
from random import randint
import shlex

# Ensure local imports work when running as a script
sys.path.append(os.path.dirname(__file__))
from cvpr_compare import (
    cvpr_compare,
    l2_distance,
    l1_distance,
    chi_square_distance,
    histogram_intersection,
)

# Defaults
IMAGE_FOLDER = os.path.join('MSRC_ObjCategImageDatabase_v2', 'Images')
DESCRIPTOR_FOLDER = 'descriptor'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto/rgb8'  # default; override via --desc-subfolder


def _load_descriptors(desc_dir: str):
    ALLFEAT = []
    ALLFILES = []
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
        ALLFEAT.append(F)
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
    return np.array(ALLFEAT), ALLFILES


def _make_montage(images, cols=5, scale=0.5, pad=2):
    if not images:
        return None
    h0, w0 = images[0].shape[:2]
    W = int(w0 * scale)
    H = int(h0 * scale)
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * (H + pad) + pad, cols * (W + pad) + pad, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        thumb = cv2.resize(img, (W, H))
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = thumb
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Visual search using precomputed descriptors')
    parser.add_argument('--desc-subfolder', default=DESCRIPTOR_SUBFOLDER, help='Subfolder inside descriptor/')
    parser.add_argument('--topk', type=int, default=15, help='Number of results to show/save')
    parser.add_argument('--results-dir', default='results', help='Base folder to save query results')
    parser.add_argument('--no-show', action='store_true', help='Do not open image windows')
    parser.add_argument('--query', default='10_10_s.bmp', help='Query filename under Images (default: 10_10_s.bmp)')
    parser.add_argument('--random-query', action='store_true', help='Pick a random query instead of using --query')
    parser.add_argument('--eval', action='store_true', help='Also run evaluation (no mAP) and save metrics')
    parser.add_argument('--eval-map', action='store_true', help='Run evaluation and include mAP (slower)')
    parser.add_argument('--vs-metric', choices=['l2', 'l1', 'chi2', 'histint', 'mahal'], default='l2', help='Distance metric for ranking')
    parser.add_argument('--eval-metric', choices=['l2', 'l1', 'chi2', 'histint', 'mahal'], default='l2', help='Distance metric for evaluation')
    parser.add_argument('--mahal-lambda', type=float, default=1e-6, help='Regularization lambda for Mahalanobis metric')
    parser.add_argument('--plot-query', default='10_10_s.bmp', help='Also save PR curve for this specific filename during evaluation')
    parser.add_argument('--run-id', default=None, help='Attach a run-id to outputs for traceability')
    args = parser.parse_args()

    desc_dir = os.path.join(DESCRIPTOR_FOLDER, args.desc_subfolder)
    ALLFEAT, ALLFILES = _load_descriptors(desc_dir)
    NIMG = ALLFEAT.shape[0]
    if NIMG == 0:
        raise SystemExit(f"No descriptors found in {desc_dir}")

    # Select query index: default specific filename, optional random
    if args.random_query:
        queryimg = randint(0, NIMG - 1)
        qname = ALLFILES[queryimg]
    else:
        # find the first index where basename matches args.query
        try:
            queryimg = next(i for i, f in enumerate(ALLFILES) if os.path.basename(f) == os.path.basename(args.query))
            qname = ALLFILES[queryimg]
        except StopIteration:
            print(f"[WARN] Query '{args.query}' not found in descriptors; using random.")
            queryimg = randint(0, NIMG - 1)
            qname = ALLFILES[queryimg]
    query = ALLFEAT[queryimg]
    
    # Choose distance metric for ranking
    def _metric_fn(name):
        if name == 'l1':
            return l1_distance
        if name == 'chi2':
            return chi_square_distance
        if name == 'histint':
            return histogram_intersection
        return l2_distance
    mfn = _metric_fn(args.vs_metric)
    dst = []
    if args.vs_metric == 'mahal':
        # Compute whitening transform and transform all features
        X = ALLFEAT.astype(np.float64)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        C = np.cov(Xc, rowvar=False)
        lam = float(args.mahal_lambda)
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 0.0, None)
        denom = np.sqrt(vals + lam)
        W = (vecs / denom)
        Y = Xc @ W
        qy = Y[queryimg]
        dists = np.linalg.norm(Y - qy, axis=1)
        for i in range(NIMG):
            dst.append((float(dists[i]), i))
    else:
        for i in range(NIMG):
            candidate = ALLFEAT[i]
            distance = mfn(query, candidate)
            dst.append((distance, i))
    dst.sort(key=lambda x: x[0])

    # Show/save top results
    SHOW = min(args.topk, NIMG)
    show_imgs = []
    for rank in range(SHOW):
        idx = dst[rank][1]
        img_path = os.path.join(IMAGE_FOLDER, ALLFILES[idx])
        img = cv2.imread(img_path)
        if img is None:
            continue
        show_imgs.append(img)
        if not args.no_show:
            disp = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
            cv2.imshow(f"{rank+1}: {ALLFILES[idx]}", disp)
            cv2.waitKey(0)
    if not args.no_show:
        cv2.destroyAllWindows()

    # Save results to disk
    now = datetime.now()
    ts = now.strftime('%m%d-%H%M')
    ts_full = now.strftime('%m%d%H%M%S')
    # Distinct folder: seconds + optional run-id suffix
    suffix = ("_" + args.run_id) if args.run_id else ""
    run_dir = os.path.join(args.results_dir, 'search', args.desc_subfolder, ts_full + suffix)
    os.makedirs(run_dir, exist_ok=True)

    # Save command used to run
    try:
        script_rel = os.path.relpath(__file__, start=os.getcwd())
    except Exception:
        script_rel = __file__
    run_cmd = "python " + script_rel + " " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    try:
        with open(os.path.join(run_dir, 'command.txt'), 'w') as f:
            f.write(run_cmd + "\n")
    except Exception:
        pass

    # Save query image
    # qname already set when selecting query
    qpath = os.path.join(IMAGE_FOLDER, qname)
    qimg = cv2.imread(qpath)
    if qimg is not None:
        cv2.imwrite(os.path.join(run_dir, 'query.jpg'), qimg)

    # Save montage of top-K
    montage = _make_montage(show_imgs, cols=5, scale=0.5)
    if montage is not None:
        cv2.imwrite(os.path.join(run_dir, f'top{SHOW}_grid.jpg'), montage)

    # Save CSV of results (rank, distance, filename)
    with open(os.path.join(run_dir, f'top{SHOW}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'distance', 'filename'])
        for r in range(SHOW):
            w.writerow([r + 1, float(dst[r][0]), ALLFILES[dst[r][1]]])

    print(f"Saved results to {run_dir}")

    # Optional evaluation in one go
    if args.eval or args.eval_map:
        # Import here to avoid circulars on module load
        from evaluation import evaluate, _save_eval
        res = evaluate(
            ALLFEAT,
            ALLFILES,
            topk=args.topk,
            metric=args.eval_metric,
            compute_map=args.eval_map,
            mahal_lambda=args.mahal_lambda,
            plot_query=args.plot_query,
        )
        _save_eval(args.results_dir, args.desc_subfolder, res, metric=args.eval_metric, topk=args.topk, run_command=run_cmd, run_id=args.run_id)


if __name__ == '__main__':
    main()

# save results to file
