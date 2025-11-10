import os
import sys
import argparse
import shlex
import numpy as np
import scipy.io as sio

def _load_descriptors(desc_dir):
    X = []
    files = []
    for name in sorted(os.listdir(desc_dir)):
        if not name.endswith('.mat'):
            continue
        path = os.path.join(desc_dir, name)
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
            base = os.path.splitext(name)[0] + '.bmp'
        files.append(str(base))
    if not X:
        raise SystemExit(f"No descriptors found in {desc_dir}")
    return np.vstack(X), files


def _fit_pca(X, dim=None, var=None, whiten=False):
    # Center
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Explained variance
    n = X.shape[0]
    ev = (S ** 2) / max(n - 1, 1)
    ev_ratio = ev / (ev.sum() + 1e-12)
    # Choose dimension
    if var is not None:
        csum = np.cumsum(ev_ratio)
        d = int(np.searchsorted(csum, float(var), side='right') + 1)
    else:
        d = int(dim) if dim is not None else min(128, Vt.shape[0])
    d = max(1, min(d, Vt.shape[0]))
    comps = Vt[:d, :]
    ev_d = ev[:d]
    # Transform
    Z = Xc @ comps.T
    if whiten:
        Z = Z / (np.sqrt(ev_d + 1e-12))
    return Z, mu, comps, ev_d, ev_ratio, d, bool(whiten)


def main():
    parser = argparse.ArgumentParser(description='Fit PCA on descriptors and save reduced features')
    parser.add_argument('--desc-subfolder', required=True, help='Input subfolder under descriptor/')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dim', type=int, default=128, help='Target dimension')
    group.add_argument('--var', type=float, default=None, help='Target explained variance ratio (0-1)')
    parser.add_argument('--whiten', action='store_true', help='Whiten by dividing by sqrt(eigenvalues)')
    parser.add_argument('--results-dir', default='descriptor', help='Base descriptor folder')
    args = parser.parse_args()

    in_dir = os.path.join(args.results_dir, args.desc_subfolder)
    X, files = _load_descriptors(in_dir)

    Z, mu, comps, ev_d, ev_ratio, d, whiten = _fit_pca(X, dim=args.dim, var=args.var, whiten=args.whiten)

    out_sub = f"{args.desc_subfolder}_pca{d}"
    out_dir = os.path.join(args.results_dir, out_sub)
    os.makedirs(out_dir, exist_ok=True)

    # Save model
    np.savez(os.path.join(out_dir, 'pca_model.npz'),
             mean=mu, components=comps, explained_variance=ev_d,
             explained_variance_ratio=ev_ratio, dim=d, whiten=whiten,
             source=args.desc_subfolder)

    # Save transformed descriptors back to .mat files
    for i, fname in enumerate(files):
        base = os.path.splitext(os.path.basename(fname))[0]
        path = os.path.join(out_dir, base + '.mat')
        params = dict(type='pca', dim=d, whiten=whiten, source=args.desc_subfolder)
        sio.savemat(path, {'F': Z[i:i+1, :], 'file': fname, 'params': params})

    # Save command used to run
    try:
        cmd = 'python ' + os.path.relpath(__file__, start=os.getcwd()) + ' ' + ' '.join(shlex.quote(a) for a in sys.argv[1:])
        with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
            f.write(cmd + '\n')
    except Exception:
        pass

    print(f"Saved PCA descriptors to {out_dir} (dim={d}, whiten={whiten})")


if __name__ == '__main__':
    main()
