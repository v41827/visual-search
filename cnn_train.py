#!/usr/bin/env python3
"""
CNN-based image retrieval on MSRC images.

Features
- Trains ResNet-18 → 128-D L2-normalised embedding with CE + label smoothing.
- Extracts embeddings for all images and evaluates with L2 and Mahalanobis
  using the existing evaluation helpers (PR curve, Top-1/Top-5, mAP, CM).
- Exports descriptors as .mat files under descriptor/cnn/resnet18_embed128
  so CLI tools (e.g., cvpr_visualsearch.py) work directly.
- Saves visual search grids for a demo query (default 10_10_s.bmp) under
  results/search/cnn/resnet18_embed128/<timestamp_metric>/.

Run
  python cnn_train.py                      # train, export descriptors, eval L2+mahal, save search grids
  python cnn_train.py --no-pretrained      # if weights cannot be downloaded
  python cnn_train.py --epochs 30 --batch-size 32
  python cnn_train.py --export-mat 0       # skip exporting .mat descriptors

Requirements: torch, torchvision, numpy, pillow, scipy
"""

import os
import sys
import csv
import math
import time
import random
import argparse
from datetime import datetime
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm

import scipy.io as sio


# Allow importing helpers from Skeleton_Python/
sys.path.append('Skeleton_Python')
from evaluation import evaluate, _save_eval  # type: ignore


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        if hasattr(torch.backends, 'mps'):
            torch.mps.manual_seed(seed)  # type: ignore
    except Exception:
        pass


def parse_label_from_name(name: str) -> int:
    try:
        return int(os.path.basename(name).split('_')[0])
    except Exception:
        return -1


def list_images(root: str) -> List[str]:
    files = []
    for fn in sorted(os.listdir(root)):
        if fn.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
            files.append(fn)
    return files


class MSRCDataset(Dataset):
    def __init__(self, root: str, files: List[str], labels: List[int], transform=None):
        self.root = root
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        path = os.path.join(self.root, fn)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx]), fn


def stratified_split(files: List[str], y_raw: List[int], val_ratio=0.2):
    uniq = sorted(set(y_raw))
    lab2idx = {c: i for i, c in enumerate(uniq)}
    y = [lab2idx[c] for c in y_raw]
    train_files, val_files, y_train, y_val = [], [], [], []
    buckets = {}
    for f, r, i in zip(files, y_raw, y):
        buckets.setdefault(i, []).append((f, r, i))
    for i, items in buckets.items():
        n = len(items)
        k = max(1, int(round(n * val_ratio)))
        random.shuffle(items)
        val = items[:k]
        tr = items[k:]
        for f, r, i in tr:
            train_files.append(f)
            y_train.append(i)
        for f, r, i in val:
            val_files.append(f)
            y_val.append(i)
    return (train_files, y_train, val_files, y_val, lab2idx)


class EmbedNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        # Try new torchvision API; fall back if unavailable or download blocked
        weights = None
        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                # Older torchvision or no weights available
                weights = None
        try:
            self.backbone = tvm.resnet18(weights=weights)
            self.pretrained_loaded = weights is not None
        except Exception:
            # As a final fallback
            self.backbone = tvm.resnet18(weights=None)
            self.pretrained_loaded = False
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embed = nn.Linear(feat_dim, emb_dim)
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_embedding: bool = True):
        f = self.backbone(x)
        z = self.embed(f)
        # L2-normalize embeddings for retrieval geometry
        e = torch.nn.functional.normalize(z, dim=1)
        logits = self.cls(e)
        return e if return_embedding else z, logits


def default_normalize_transform():
    try:
        from torchvision.models import ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1
        mean = tuple(w.meta.get('mean', (0.485, 0.456, 0.406)))
        std = tuple(w.meta.get('std', (0.229, 0.224, 0.225)))
    except Exception:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    return T.Normalize(mean=mean, std=std)


def save_search_grid(img_root: str, files: List[str], dists: np.ndarray, order: List[int], qidx: int,
                     out_dir: str, topk: int = 15):
    os.makedirs(out_dir, exist_ok=True)
    # Save query
    Image.open(os.path.join(img_root, files[qidx])).convert('RGB').save(os.path.join(out_dir, 'query.jpg'))
    # Save grid
    k = min(topk, len(order))
    imgs = [Image.open(os.path.join(img_root, files[i])).convert('RGB') for i in order[:k]]
    if imgs:
        # Simple montage
        w0, h0 = imgs[0].size
        cols = 5
        scale = 0.5
        pad = 2
        W, H = int(w0 * scale), int(h0 * scale)
        rows = (k + cols - 1) // cols
        canvas = Image.new('RGB', (cols * (W + pad) + pad, rows * (H + pad) + pad), (255, 255, 255))
        for idx, im in enumerate(imgs):
            r, c = divmod(idx, cols)
            thumb = im.resize((W, H))
            x0 = pad + c * (W + pad)
            y0 = pad + r * (H + pad)
            canvas.paste(thumb, (x0, y0))
        canvas.save(os.path.join(out_dir, f'top{k}_grid.jpg'), quality=95)
    # Save CSV
    with open(os.path.join(out_dir, f'top{topk}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'distance', 'filename'])
        for r, i in enumerate(order[:k], start=1):
            w.writerow([r, float(dists[i]), files[i]])


def main():
    ap = argparse.ArgumentParser(description='Train CNN embedding and export descriptors + eval/search (MSRC)')
    ap.add_argument('--img-root', default=os.path.join('MSRC_ObjCategImageDatabase_v2', 'Images'))
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--desc-subfolder', default='cnn/resnet18_embed128')
    ap.add_argument('--export-mat', type=int, default=1, help='Export .mat descriptors under descriptor/<subfolder> (1/0)')
    ap.add_argument('--models-dir', default='results/models', help='Base folder to save/load model checkpoints')
    ap.add_argument('--ckpt-out', default='', help='Path to save best checkpoint (.pt). If empty, auto under results/models/...')
    ap.add_argument('--resume', default='', help='Path to checkpoint to resume/evaluate (.pt)')
    ap.add_argument('--export-only', action='store_true', help='Skip training; only export/eval/search using --resume')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--label-smooth', type=float, default=0.1)
    ap.add_argument('--lr-backbone', type=float, default=1e-4)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--query', default='10_10_s.bmp')
    ap.add_argument('--topk', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print('Device:', device)

    # Data
    files = list_images(args.img_root)
    if not files:
        raise SystemExit(f'No images found in {args.img_root}')
    y_raw = [parse_label_from_name(f) for f in files]
    train_files, y_train, val_files, y_val, lab2idx = stratified_split(files, y_raw, val_ratio=0.2)
    num_classes = len(lab2idx)
    print(f'Total {len(files)} images | Train {len(train_files)} | Val {len(val_files)} | Classes {num_classes}')

    norm = default_normalize_transform()
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        norm,
    ])
    eval_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        norm,
    ])

    ds_tr = MSRCDataset(args.img_root, train_files, y_train, transform=train_tf)
    ds_va = MSRCDataset(args.img_root, val_files, y_val, transform=eval_tf)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = EmbedNet(num_classes=num_classes, emb_dim=args.emb_dim, pretrained=(not args.no_pretrained))
    print('Pretrained weights loaded:', getattr(model, 'pretrained_loaded', False))
    model.to(device)

    # Optional: load checkpoint before training/export
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded checkpoint from {args.resume}")

    # Optimiser & loss
    params = [
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': args.lr_backbone},
        {'params': list(model.embed.parameters()) + list(model.cls.parameters()), 'lr': args.lr_head},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=float(args.label_smooth))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Train
    def run_epoch(dl, train: bool):
        model.train() if train else model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb, _ in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                emb, logits = model(xb, return_embedding=True)
                loss = crit(logits, yb)
                if train:
                    loss.backward()
                    opt.step()
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
        return loss_sum / max(1, total), correct / max(1, total)

    best_val, best_state = -1.0, None
    if not args.export_only:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(dl_tr, True)
            va_loss, va_acc = run_epoch(dl_va, False)
            sched.step()
            if va_acc > best_val:
                best_val = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == 1:
                print(f'Epoch {epoch:02d}/{args.epochs} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}')
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        print('Best val acc:', best_val)
        # Save checkpoint
        ts_full = datetime.now().strftime('%m%d%H%M%S')
        if not args.ckpt_out:
            ckpt_dir = os.path.join(args.models_dir, args.desc_subfolder)
            os.makedirs(ckpt_dir, exist_ok=True)
            args.ckpt_out = os.path.join(ckpt_dir, f'{ts_full}.pt')
        meta = {
            'desc_subfolder': args.desc_subfolder,
            'emb_dim': int(args.emb_dim),
            'classes': sorted(list(lab2idx.keys())),
            'label_map': lab2idx,
            'pretrained': bool(not args.no_pretrained),
            'best_val_acc': float(best_val),
        }
        torch.save({'state_dict': model.state_dict(), 'meta': meta}, args.ckpt_out)
        print('Saved checkpoint to', args.ckpt_out)
    else:
        if not args.resume:
            raise SystemExit('--export-only requires --resume to load weights')
        model.eval()

    # Extract embeddings on full set
    full_ds = MSRCDataset(args.img_root, files, [lab2idx[parse_label_from_name(f)] for f in files], transform=eval_tf)
    full_dl = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    feats_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, fb in full_dl:
            xb = xb.to(device)
            emb, _ = model(xb, return_embedding=True)
            feats_list.append(emb.detach().cpu().numpy())
    FEATS = np.concatenate(feats_list, axis=0).astype(np.float64)
    # Ensure strict L2 norm
    FEATS /= np.clip(np.linalg.norm(FEATS, axis=1, keepdims=True), 1e-12, None)
    FILES = list(files)

    # Export descriptors as .mat so CLI tools can use them
    if args.export_mat:
        out_dir = os.path.join('descriptor', args.desc_subfolder)
        os.makedirs(out_dir, exist_ok=True)
        params = {
            'type': 'cnn',
            'arch': 'resnet18',
            'embedding': int(args.emb_dim),
            'pretrained': bool(not args.no_pretrained and getattr(model, 'pretrained_loaded', False)),
            'normalize': 'l2',
        }
        for i, fname in enumerate(FILES):
            base = os.path.splitext(os.path.basename(fname))[0]
            fout = os.path.join(out_dir, base + '.mat')
            sio.savemat(fout, {'F': FEATS[i:i+1, :], 'file': fname, 'params': params})
        print(f'Saved CNN descriptors to {out_dir}')

    # Evaluate with existing helpers and save outputs (L2 and Mahalanobis)
    for metric in ['l2', 'mahal']:
        res = evaluate(FEATS, FILES, topk=int(args.topk), metric=metric, compute_map=True, mahal_lambda=1e-6, plot_query=args.query)
        _save_eval(args.results_dir, args.desc_subfolder, res, metric=metric, topk=int(args.topk), run_command='python cnn_train.py', run_id=f'resnet18_{metric}')
        print('Saved eval for', metric)
        # Also save a visual search grid for the demo query
        # Rank by metric
        qname = os.path.basename(args.query)
        try:
            qidx = next(i for i, f in enumerate(FILES) if os.path.basename(f) == qname)
        except StopIteration:
            print(f"[WARN] Query {qname} not found; skipping grid for metric {metric}")
            continue
        X = FEATS.astype(np.float64)
        if metric == 'mahal':
            mu = X.mean(axis=0, keepdims=True)
            Xc = X - mu
            C = np.cov(Xc, rowvar=False)
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 0.0, None)
            W = (vecs / np.sqrt(vals + 1e-6))
            Y = Xc @ W
            qv = Y[qidx]
            d = np.linalg.norm(Y - qv, axis=1)
        else:
            qv = X[qidx]
            d = np.linalg.norm(X - qv, axis=1)
        order = np.argsort(d)
        order = [i for i in order if i != qidx]
        ts_full = datetime.now().strftime('%m%d%H%M%S')
        search_dir = os.path.join(args.results_dir, 'search', args.desc_subfolder, f'{ts_full}_{metric}')
        save_search_grid(args.img_root, FILES, d, order, qidx, search_dir, topk=int(args.topk))
        print('Saved search to', search_dir)

    print('Done.')


if __name__ == '__main__':
    main()
