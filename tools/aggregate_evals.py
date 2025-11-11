#!/usr/bin/env python3
import os
import csv
import re
import argparse


def parse_desc(sub: str):
    # Returns a dict of parsed fields from desc_subfolder
    out = {
        'descriptor': '', 'grid': '', 'colorspace': '', 'color_bins': '',
        'texture': '', 'orient_bins': '', 'signed': '', 'lbp_points': '', 'lbp_radius': '',
        'pca_dim': '', 'detector': '', 'codebook': ''
    }
    s = sub.strip('/')
    if s.startswith('globalRGBhisto/'):
        out['descriptor'] = 'hist'
        m = re.search(r'rgb(\d+)(?:_pca(\d+))?$', s)
        if m:
            out['color_bins'] = m.group(1)
            if m.group(2):
                out['pca_dim'] = m.group(2)
    elif s.startswith('spatialGrid/'):
        out['descriptor'] = 'spatial'
        tail = s.split('/', 1)[1]
        # e.g., 2x2_rgb_c8_grad16u[_pca128]
        pca = re.search(r'_pca(\d+)$', tail)
        if pca:
            out['pca_dim'] = pca.group(1)
            tail = tail[:pca.start()]
        m = re.match(r'([^_]+)_([^_]+)_c(\d+)_(.+)$', tail)
        if m:
            out['grid'] = m.group(1)
            out['colorspace'] = m.group(2)
            out['color_bins'] = m.group(3)
            tex = m.group(4)
            if tex.startswith('grad'):
                out['texture'] = 'grad'
                m2 = re.match(r'grad(\d+)([us])$', tex)
                if m2:
                    out['orient_bins'] = m2.group(1)
                    out['signed'] = 'signed' if m2.group(2) == 's' else 'unsigned'
            elif tex.startswith('lbp'):
                out['texture'] = 'lbp'
                m2 = re.match(r'lbp(\d+)r(\d+)$', tex)
                if m2:
                    out['lbp_points'] = m2.group(1)
                    out['lbp_radius'] = m2.group(2)
            elif tex == 'colorOnly':
                out['texture'] = 'none'
    elif s.startswith('BoVW/'):
        out['descriptor'] = 'bovw'
        tail = s.split('/', 1)[1]
        # e.g., sift_k400[_pca128]
        pca = re.search(r'_pca(\d+)$', tail)
        if pca:
            out['pca_dim'] = pca.group(1)
            tail = tail[:pca.start()]
        m = re.match(r'([^_]+)_k(\d+)$', tail)
        if m:
            out['detector'] = m.group(1)
            out['codebook'] = m.group(2)
    return out


def read_summary(path):
    vals = {}
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            k, v = row[0], row[1]
            vals[k] = v
    return vals


def main():
    ap = argparse.ArgumentParser(description='Aggregate evaluation summaries into one CSV')
    ap.add_argument('--eval-root', default='results/eval', help='Root folder containing eval runs')
    ap.add_argument('--out', default='results/eval/REPORT_summary.csv', help='Output CSV path')
    ap.add_argument('--run-id', default=None, help='Filter rows to a specific run_id')
    args = ap.parse_args()

    rows = []
    for root, dirs, files in os.walk(args.eval_root):
        if 'summary.csv' in files:
            sfile = os.path.join(root, 'summary.csv')
            vals = read_summary(sfile)
            sub = vals.get('desc_subfolder') or os.path.relpath(root, args.eval_root)
            parsed = parse_desc(sub)
            # Timestamp: prefer summary, else take from folder name
            ts = vals.get('timestamp', '')
            ts_full = vals.get('timestamp_full', '')
            if not ts:
                ts = os.path.basename(root)
            if not ts_full:
                ts_full = re.sub(r'\D', '', ts)  # strip non-digits
            row = {
                'subfolder': sub,
                'descriptor': parsed['descriptor'],
                'grid': parsed['grid'],
                'colorspace': parsed['colorspace'],
                'color_bins': parsed['color_bins'],
                'texture': parsed['texture'],
                'orient_bins': parsed['orient_bins'],
                'signed': parsed['signed'],
                'lbp_points': parsed['lbp_points'],
                'lbp_radius': parsed['lbp_radius'],
                'detector': parsed['detector'],
                'codebook': parsed['codebook'],
                'pca_dim': parsed['pca_dim'],
                'distance': vals.get('metric_name', ''),
                'run_ts': ts,
                'run_ts_full': ts_full,
                'run_id': vals.get('run_id', ''),
                'top1_acc': vals.get('top1_acc', ''),
                'top5_acc': vals.get('top5_acc', ''),
                'mean_P_at_15': vals.get('mean_P_at_15', ''),
                'mean_R_at_15': vals.get('mean_R_at_15', ''),
                'mAP': vals.get('mAP', ''),
                'N': vals.get('N', ''),
                'topk': vals.get('topk', ''),
            }
            if args.run_id is None or row['run_id'] == args.run_id:
                rows.append(row)

    # Reorder columns so key metrics come first, followed by the rest
    cols_metrics_first = ['subfolder', 'distance', 'top1_acc', 'top5_acc', 'mean_P_at_15', 'mean_R_at_15', 'mAP']
    cols_rest = ['descriptor', 'grid', 'colorspace', 'color_bins', 'texture', 'orient_bins', 'signed',
                 'lbp_points', 'lbp_radius', 'detector', 'codebook', 'pca_dim', 'run_ts', 'run_ts_full', 'run_id',
                 'topk', 'N']
    cols = cols_metrics_first + cols_rest
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == '__main__':
    main()
