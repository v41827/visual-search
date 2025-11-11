import os
import sys
import csv
import argparse
from datetime import datetime
from typing import List, Tuple

import numpy as np
import scipy.io as sio


def _load_descriptors(desc_dir: str) -> Tuple[np.ndarray, List[str]]:
    X = []
    files: List[str] = []
    for filename in sorted(os.listdir(desc_dir)):
        if not filename.endswith('.mat'):
            continue
        path = os.path.join(desc_dir, filename)
        try:
            mat = sio.loadmat(path)
        except Exception:
            continue
        F = mat.get('F')
        if F is None:
            continue
        f = np.asarray(F).reshape(-1).astype(np.float64)
        X.append(f)
        base = mat.get('file')
        if isinstance(base, np.ndarray):
            try:
                base = str(base.squeeze())
            except Exception:
                base = None
        if not base:
            base = os.path.splitext(filename)[0] + '.bmp'
        files.append(str(base))
    if not X:
        raise SystemExit(f"No descriptors found in {desc_dir}")
    return np.vstack(X), files


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


def _descriptor_label(desc_subfolder: str) -> str:
    s = desc_subfolder.replace('\\', '/').strip('/')
    low = s.lower()
    import re
    m = re.match(r'^globalrgbhisto/rgb(\d+)$', low)
    if m:
        return f"hist_rgb_{m.group(1)}bins"
    if low.startswith('spatialgrid/'):
        return 'spatial_grid_' + low.split('/', 1)[1].replace('/', '_')
    if low.startswith('bovw/'):
        return 'bow_' + low.split('/', 1)[1].replace('/', '_')
    if low.startswith('cnn/'):
        return low.replace('/', '_')
    return low.replace('/', '_')


def _save_results(out_dir: str, y_true: np.ndarray, y_pred: np.ndarray, files: List[str], summary: dict, prefix: str = "", descriptor_label: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    # command
    try:
        cmd = 'python ' + os.path.relpath(__file__, start=os.getcwd()) + ' ' + ' '.join(sys.argv[1:])
        with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
            f.write(cmd + '\n')
    except Exception:
        pass

    # per-sample
    suffix = f"_{prefix}" if prefix else ""
    with open(os.path.join(out_dir, f'per_sample{suffix}.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['file', 'true', 'pred', 'correct'])
        for i in range(len(files)):
            w.writerow([files[i], int(y_true[i]), int(y_pred[i]), int(y_true[i] == y_pred[i])])

    # summary
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['metric', 'value'])
        for k, v in summary.items():
            w.writerow([k, v])

    # confusion matrix
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except Exception:
            sns = None

        labels_sorted = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

        # save CSV
        with open(os.path.join(out_dir, f'confusion_matrix{suffix}.csv'), 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow([''] + list(map(int, labels_sorted)))
            for i, lab in enumerate(labels_sorted):
                w.writerow([int(lab)] + list(map(int, cm[i].tolist())))

        # matplotlib plot with annotations
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        title = 'SVM Confusion Matrix — ' + (descriptor_label or 'descriptor')
        if prefix:
            title += f' ({prefix})'
        plt.title(title)
        plt.colorbar()
        plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=90)
        plt.yticks(range(len(labels_sorted)), labels_sorted)
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(int(cm[i, j])), ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'confusion_matrix{suffix}.png'))
        plt.close()

        # seaborn heatmap
        if sns is not None:
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                             xticklabels=labels_sorted, yticklabels=labels_sorted, linewidths=0.5, linecolor='white', square=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            title = 'SVM Confusion Matrix — ' + (descriptor_label or 'descriptor')
            if prefix:
                title += f' ({prefix})'
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'confusion_matrix_heatmap{suffix}.png'))
            plt.close()
    except Exception:
        pass


def _save_per_class(out_dir: str, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""):
    """Save per-class precision/recall/F1/support as a CSV.
    Prefix can be 'oof', 'train', or 'test' to distinguish contexts.
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except Exception:
        return

    labs = np.unique(np.concatenate([y_true, y_pred]))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labs, zero_division=0)
    suffix = f"_{prefix}" if prefix else ""
    path = os.path.join(out_dir, f"per_class{suffix}.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['class', 'support', 'precision', 'recall', 'f1'])
        for i, c in enumerate(labs):
            w.writerow([int(c), int(sup[i]), float(p[i]), float(r[i]), float(f1[i])])


def main():
    parser = argparse.ArgumentParser(description='Train/test an SVM classifier on descriptors')
    parser.add_argument('--desc-subfolder', required=True, help='Subfolder under descriptor/, e.g., spatialGrid/2x2_rgb_c8_grad16u')
    parser.add_argument('--results-dir', default='results', help='Base folder to save SVM results')
    parser.add_argument('--kernel', choices=['linear', 'rbf', 'poly'], default='linear')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--gamma', default='scale', help="'scale', 'auto', or float for RBF/poly")
    parser.add_argument('--degree', type=int, default=3, help='Degree for poly kernel')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction for test split')
    parser.add_argument('--cv', type=int, default=0, help='Stratified K-folds; 0 disables CV and uses a single split')
    parser.add_argument('--standardize', action='store_true', help='Standardize features (z-score)')
    parser.add_argument('--random-state', type=int, default=42)

    args = parser.parse_args()

    # Lazy import scikit-learn with friendly message
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import StratifiedKFold, train_test_split
    except Exception as e:
        raise SystemExit('scikit-learn is required for SVM. Please install scikit-learn.')

    X, files = _load_descriptors(os.path.join('descriptor', args.desc_subfolder))
    y = _labels_from_filenames(files)

    # Build classifier
    gamma_val = args.gamma
    if isinstance(gamma_val, str):
        if gamma_val not in ('scale', 'auto'):
            try:
                gamma_val = float(gamma_val)
            except Exception:
                gamma_val = 'scale'

    clf = SVC(kernel=args.kernel, C=args.C, gamma=gamma_val, degree=args.degree, random_state=args.random_state)
    if args.standardize:
        model = make_pipeline(StandardScaler(with_mean=True), clf)
    else:
        model = clf

    ts = datetime.now().strftime('%m%d-%H%M')
    out_dir = os.path.join(args.results_dir, 'svm', args.desc_subfolder.replace('/', '_'), ts)

    if args.cv and args.cv > 1:
        skf = StratifiedKFold(n_splits=int(args.cv), shuffle=True, random_state=args.random_state)
        accs = []
        f1s = []
        fold = 0
        y_pred_oof = np.full(y.shape, fill_value=np.nan)
        for train_idx, test_idx in skf.split(X, y):
            fold += 1
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, average='macro'))
            y_pred_oof[test_idx] = y_pred
        summary = {
            'cv_folds': int(args.cv),
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s)),
            'N': int(X.shape[0]),
        }
        # Save out-of-fold (OOF) predictions and confusion matrix (test-like view across folds)
        from sklearn.metrics import accuracy_score as _acc, f1_score as _f1
        oof_mask = ~np.isnan(y_pred_oof)
        if np.any(oof_mask):
            y_pred_oof = y_pred_oof.astype(int)
            oof_acc = float(_acc(y[oof_mask], y_pred_oof[oof_mask]))
            oof_f1 = float(_f1(y[oof_mask], y_pred_oof[oof_mask], average='macro'))
            summary.update({'oof_accuracy': oof_acc, 'oof_macro_f1': oof_f1})
        label = _descriptor_label(args.desc_subfolder)
        _save_results(out_dir, y[oof_mask], y_pred_oof[oof_mask], [files[i] for i in np.where(oof_mask)[0]], summary, prefix='oof', descriptor_label=label)
        _save_per_class(out_dir, y[oof_mask], y_pred_oof[oof_mask], prefix='oof')

        # Also train on full data and save the (optimistic) training CM for reference
        model.fit(X, y)
        y_pred_train = model.predict(X)
        _save_results(out_dir, y, y_pred_train, files, summary, prefix='train', descriptor_label=label)
        _save_per_class(out_dir, y, y_pred_train, prefix='train')
        print(f"Saved SVM CV results to {out_dir}")
    else:
        Xtr, Xte, ytr, yte, ftr, fte = train_test_split(X, y, files, test_size=args.test_size, stratify=y, random_state=args.random_state)
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        acc = float(accuracy_score(yte, y_pred))
        f1m = float(f1_score(yte, y_pred, average='macro'))
        summary = {
            'kernel': args.kernel,
            'C': args.C,
            'gamma': gamma_val,
            'degree': args.degree,
            'standardize': bool(args.standardize),
            'test_size': args.test_size,
            'accuracy': acc,
            'macro_f1': f1m,
            'N_train': int(Xtr.shape[0]),
            'N_test': int(Xte.shape[0]),
        }
        label = _descriptor_label(args.desc_subfolder)
        _save_results(out_dir, np.array(yte), np.array(y_pred), fte, summary, prefix='test', descriptor_label=label)
        _save_per_class(out_dir, np.array(yte), np.array(y_pred), prefix='test')
        print(f"Saved SVM results to {out_dir} (accuracy={acc:.4f})")


if __name__ == '__main__':
    main()
