#!/usr/bin/env python3
"""
Scratch CNN for image retrieval on MSRC images.

Architecture (Conv4):
  Conv32→BN→ReLU→MaxPool → Conv64→BN→ReLU→MaxPool →
  Conv128→BN→ReLU→MaxPool → Conv256→BN→ReLU → GAP → Dropout(0.3) →
  FC(256→128) + L2 → FC(128→num_classes)

Training:
  - Stage 1: Cross-Entropy with label smoothing (default 80 epochs)
  - Optional Stage 2: Triplet (batch-hard within batch) to refine embedding

Outputs:
  - .mat descriptors under descriptor/cnn/conv_scratch_embed128
  - Evaluation (L2 and Mahalanobis) under results/eval/cnn/conv_scratch_embed128/<timestamp>
  - Visual search grid for query (default 10_10_s.bmp) under
    results/search/cnn/conv_scratch_embed128/<timestamp_metric>/
  - Checkpoints under results/models/cnn/conv_scratch_embed128/<timestamp>.pt

Run examples:
  python cnn_scratch_train.py
  python cnn_scratch_train.py --epochs 80 --with-triplet --triplet-epochs 15
  python cnn_scratch_train.py --resume results/models/cnn/conv_scratch_embed128/0123123456.pt --export-only
"""

import os
import sys
import csv
import math
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

import scipy.io as sio

# Allow importing helpers from Skeleton_Python
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


class Conv4Net(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc_embed = nn.Linear(256, emb_dim)
        self.fc_cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)  # (N, 256)
        x = self.dropout(x)
        z = self.fc_embed(x)
        e = nn.functional.normalize(z, dim=1)
        logits = self.fc_cls(e)
        return e, logits


def default_normalize_transform():
    # Use ImageNet-like stats
    return T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    # emb: (B,D), labels: (B,) long
    with torch.no_grad():
        # Pairwise distances (L2)
        # torch.cdist is numerically stable on MPS/CPU/GPU
        dist = torch.cdist(emb, emb, p=2)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(emb.size(0), dtype=torch.bool, device=emb.device)
        pos_mask = same & (~eye)
        neg_mask = ~same

    losses = []
    for i in range(emb.size(0)):
        pos_d = dist[i][pos_mask[i]]
        neg_d = dist[i][neg_mask[i]]
        if pos_d.numel() == 0 or neg_d.numel() == 0:
            continue
        d_ap = pos_d.max()  # hardest positive
        d_an = neg_d.min()  # hardest negative
        losses.append(nn.functional.relu(d_ap - d_an + margin))
    if not losses:
        return torch.tensor(0.0, device=emb.device)
    return torch.stack(losses).mean()


def save_search_grid(img_root: str, files: List[str], dists: np.ndarray, order: List[int], qidx: int,
                     out_dir: str, topk: int = 15):
    os.makedirs(out_dir, exist_ok=True)
    # Save query
    Image.open(os.path.join(img_root, files[qidx])).convert('RGB').save(os.path.join(out_dir, 'query.jpg'))
    # Save grid
    k = min(topk, len(order))
    imgs = [Image.open(os.path.join(img_root, files[i])).convert('RGB') for i in order[:k]]
    if imgs:
        w0, h0 = imgs[0].size
        cols, scale, pad = 5, 0.5, 2
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
    ap = argparse.ArgumentParser(description='Train scratch CNN embedding and export descriptors + eval/search (MSRC)')
    ap.add_argument('--img-root', default=os.path.join('MSRC_ObjCategImageDatabase_v2', 'Images'))
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--desc-subfolder', default='cnn/conv_scratch_embed128')
    ap.add_argument('--export-mat', type=int, default=1, help='Export .mat descriptors under descriptor/<subfolder> (1/0)')
    ap.add_argument('--models-dir', default='results/models', help='Base folder to save/load model checkpoints')
    ap.add_argument('--ckpt-out', default='', help='Path to save best checkpoint (.pt). If empty, auto under results/models/...')
    ap.add_argument('--resume', default='', help='Path to checkpoint to resume/evaluate (.pt)')
    ap.add_argument('--export-only', action='store_true', help='Skip training; only export/eval/search using --resume')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--label-smooth', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3, help='Base LR for CE stage')
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--with-triplet', action='store_true', help='Enable Triplet fine-tune stage')
    ap.add_argument('--triplet-epochs', type=int, default=15)
    ap.add_argument('--triplet-margin', type=float, default=0.2)
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
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value='random'),
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

    model = Conv4Net(num_classes=num_classes, emb_dim=args.emb_dim).to(device)

    # Stage 1: Cross-Entropy
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit_ce = nn.CrossEntropyLoss(label_smoothing=float(args.label_smooth))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def run_epoch_ce(dl, train: bool):
        model.train() if train else model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb, _ in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                emb, logits = model(xb)
                loss = crit_ce(logits, yb)
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
            tr_loss, tr_acc = run_epoch_ce(dl_tr, True)
            va_loss, va_acc = run_epoch_ce(dl_va, False)
            sched.step()
            if va_acc > best_val:
                best_val = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == 1:
                print(f'[CE] Epoch {epoch:02d}/{args.epochs} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}')

        # Optional Stage 2: Triplet fine-tune
        if args.with_triplet and args.triplet_epochs > 0:
            crit_tri_margin = float(args.triplet_margin)
            # Smaller LR for refinement
            opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=args.weight_decay)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.triplet_epochs)

            def run_epoch_tri(dl, train: bool):
                model.train() if train else model.eval()
                loss_sum, correct, total = 0.0, 0, 0
                for xb, yb, _ in dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    if train:
                        opt2.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(train):
                        emb, logits = model(xb)
                        loss = batch_hard_triplet_loss(emb, yb, margin=crit_tri_margin)
                        if train:
                            loss.backward()
                            opt2.step()
                    loss_sum += loss.item() * xb.size(0)
                    pred = logits.argmax(1)
                    correct += (pred == yb).sum().item()
                    total += xb.size(0)
                return loss_sum / max(1, total), correct / max(1, total)

            for epoch in range(1, args.triplet_epochs + 1):
                tr_loss, tr_acc = run_epoch_tri(dl_tr, True)
                va_loss, va_acc = run_epoch_tri(dl_va, False)
                sched2.step()
                # Still track best val acc (proxy); embedding quality usually improves
                if va_acc > best_val:
                    best_val = va_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch % 5 == 0 or epoch == 1:
                    print(f'[TRI] Epoch {epoch:02d}/{args.triplet_epochs} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}')

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
            'best_val_acc': float(best_val),
            'with_triplet': bool(args.with_triplet),
        }
        torch.save({'state_dict': model.state_dict(), 'meta': meta}, args.ckpt_out)
        print('Saved checkpoint to', args.ckpt_out)
    else:
        if not args.resume:
            raise SystemExit('--export-only requires --resume to load weights')
        ckpt = torch.load(args.resume, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        model.load_state_dict(sd, strict=True)
        model.eval()

    # Extract embeddings on full set
    full_ds = MSRCDataset(args.img_root, files, [lab2idx[parse_label_from_name(f)] for f in files], transform=eval_tf)
    full_dl = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    feats_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, fb in full_dl:
            xb = xb.to(device)
            emb, _ = model(xb)
            feats_list.append(emb.detach().cpu().numpy())
    FEATS = np.concatenate(feats_list, axis=0).astype(np.float64)
    FEATS /= np.clip(np.linalg.norm(FEATS, axis=1, keepdims=True), 1e-12, None)
    FILES = list(files)

    # Export descriptors
    if args.export_mat:
        out_dir = os.path.join('descriptor', args.desc_subfolder)
        os.makedirs(out_dir, exist_ok=True)
        params = {
            'type': 'cnn_scratch',
            'arch': 'conv4',
            'embedding': int(args.emb_dim),
            'normalize': 'l2',
        }
        for i, fname in enumerate(FILES):
            base = os.path.splitext(os.path.basename(fname))[0]
            fout = os.path.join(out_dir, base + '.mat')
            sio.savemat(fout, {'F': FEATS[i:i+1, :], 'file': fname, 'params': params})
        print(f'Saved CNN (scratch) descriptors to {out_dir}')

    # Evaluate (L2 and Mahalanobis)
    for metric in ['l2', 'mahal']:
        res = evaluate(FEATS, FILES, topk=int(args.topk), metric=metric, compute_map=True, mahal_lambda=1e-6, plot_query=args.query)
        _save_eval(args.results_dir, args.desc_subfolder, res, metric=metric, topk=int(args.topk), run_command='python cnn_scratch_train.py', run_id=f'conv4_{metric}')
        print('Saved eval for', metric)

        # Visual search grid (demo query)
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

