import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import cv2
import scipy.io as sio

# Ensure local imports work when running as a script
sys.path.append(os.path.dirname(__file__))
from descriptor_library import (
    extract_random,
    extract_global_color_histogram,
    extract_spatial_grid_color_texture,
    build_bovw_vocabulary,
    extract_bovw,
)


def find_images(dataset_folder: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    for name in sorted(os.listdir(dataset_folder)):
        if name.lower().endswith(exts):
            files.append(os.path.join(dataset_folder, name))
    return files


def to_rgb01(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to RGB float [0,1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)


def main():
    parser = argparse.ArgumentParser(
        description="Compute image descriptors for the MSRC (or custom) dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-folder",
        default=os.path.join("MSRC_ObjCategImageDatabase_v2", "Images"),
        help="Folder containing input images",
    )
    parser.add_argument(
        "--out-folder",
        default="descriptor",
        help="Base output folder to write descriptor .mat files",
    )
    parser.add_argument(
        "--ext",
        default=".bmp",
        help="Image file extension to search for",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only process the first N images",
    )
    # Optional PCA (applied to the output subfolder after computing this run)
    pca_grp = parser.add_mutually_exclusive_group()
    pca_grp.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="After computing, reduce features with PCA to this dimension",
    )
    pca_grp.add_argument(
        "--pca-var",
        type=float,
        default=None,
        help="After computing, reduce features with PCA to reach this explained variance (0-1)",
    )
    parser.add_argument(
        "--pca-whiten",
        action="store_true",
        help="Whiten PCA features (divide by sqrt(eigenvalues))",
    )

    subparsers = parser.add_subparsers(dest="descriptor", required=True)

    # Global color histogram
    p_glob = subparsers.add_parser("global_color_hist", help="Global 3D colour histogram (RGB only)")
    p_glob.add_argument("--bins", type=int, default=8, choices=[4, 8, 16, 32, 64], help="Bins per channel")
    p_glob.add_argument("--norm", choices=["l1", "l2", "none"], default="l1")

    # Spatial grid (colour + texture)
    p_grid = subparsers.add_parser("spatial_grid", help="Spatial grid: colour + texture (LBP or gradient)")
    p_grid.add_argument("--grid", default="2x2", help="Grid as RxC, e.g. 2x2 or 3x1")
    p_grid.add_argument("--bins", type=int, default=8, choices=[4, 8, 16, 32, 64], help="Colour bins per channel")
    p_grid.add_argument("--colorspace", choices=["rgb", "hsv"], default="rgb")
    # Texture branch selection and params
    p_grid.add_argument("--texture", choices=["none", "lbp", "grad"], default="lbp", help="Texture type")
    p_grid.add_argument("--lbp-points", type=int, default=8, help="LBP neighbors (supports 8)")
    p_grid.add_argument("--lbp-radius", type=int, default=1, help="LBP radius")
    p_grid.add_argument("--orient-bins", type=int, default=16, choices=[8, 16, 32], help="Gradient orientation bins for texture=grad")
    p_grid.add_argument("--signed", action="store_true", help="Use signed gradient (0-360) for texture=grad. Default unsigned (0-180)")
    # Mix controls
    p_grid.add_argument("--color-only", action="store_true", help="Use colour only (ignore texture)")
    p_grid.add_argument("--texture-only", action="store_true", help="Use texture only (ignore colour)")
    p_grid.add_argument("--color-weight", type=float, default=1.0, help="Weight for colour per-cell histogram")
    p_grid.add_argument("--texture-weight", type=float, default=1.0, help="Weight for texture per-cell histogram")
    p_grid.add_argument("--per-cell-normalize", action="store_true", help="L1-normalize per-cell histograms")
    p_grid.add_argument("--final-norm", choices=["l1", "l2", "none"], default="l2")

    # Bag of Visual Words (SIFT)
    p_bow = subparsers.add_parser("bow", help="Bag of Visual Words with SIFT")
    p_bow.add_argument("--codebook-size", type=int, default=200, help="Vocabulary size (K)")
    p_bow.add_argument("--detector", choices=["sift", "harris"], default="sift", help="Keypoint detector for BoVW")
    p_bow.add_argument(
        "--vocab-file",
        default=None,
        help="Path to vocabulary .npz (saved/loaded). Defaults to out-folder/vocab_k{K}.npz",
    )
    p_bow.add_argument("--build-vocab", action="store_true", help="Build the vocabulary from dataset images")
    p_bow.add_argument("--max-total-desc", type=int, default=100000, help="Max descriptors used for kmeans")
    p_bow.add_argument("--desc-per-image", type=int, default=500, help="Max descriptors per image for kmeans")
    p_bow.add_argument("--max-features", type=int, default=500, help="SIFT nfeatures cap per image")
    p_bow.add_argument("--norm", choices=["l1", "l2", "none"], default="l1")

    args = parser.parse_args()

    # Collect images
    exts = (args.ext.lower(),)
    img_files = find_images(args.dataset_folder, exts)
    if args.limit is not None:
        img_files = img_files[: args.limit]
    if not img_files:
        raise SystemExit(f"No images with extension {args.ext} in {args.dataset_folder}")

    # Output subfolder based on descriptor + params
    if args.descriptor == "global_color_hist":
        # Save to descriptor/globalRGBhisto/rgb{bins}
        subfolder = os.path.join("globalRGBhisto", f"rgb{args.bins}")
    elif args.descriptor == "spatial_grid":
        gr, gc = map(int, args.grid.lower().split("x"))
        base = f"{gr}x{gc}_{args.colorspace}_c{args.bins}"
        # Decide which branches included for naming
        include_color = not args.texture_only
        include_texture = (args.texture != "none") and (not args.color_only)
        if include_texture:
            if args.texture == "lbp":
                base += f"_lbp{args.lbp_points}r{args.lbp_radius}"
            elif args.texture == "grad":
                base += f"_grad{args.orient_bins}{'s' if args.signed else 'u'}"
        if include_color and not include_texture:
            base += "_colorOnly"
        if include_texture and not include_color:
            base += "_texOnly"
        subfolder = os.path.join("spatialGrid", base)
    else:  # bow
        # Save to descriptor/BoVW/{detector}_k{K}
        subfolder = os.path.join("BoVW", f"{args.detector}_k{args.codebook_size}")

    out_dir = os.path.join(args.out_folder, subfolder)
    os.makedirs(out_dir, exist_ok=True)

    # Precompute/load vocabulary if BOW
    vocab = None
    vocab_path = None
    if args.descriptor == "bow":
        vocab_path = args.vocab_file
        if vocab_path is None:
            # place vocab in out_folder/BoVW
            vocab_dir = os.path.join(args.out_folder, "BoVW")
            os.makedirs(vocab_dir, exist_ok=True)
            vocab_path = os.path.join(vocab_dir, f"vocab_k{args.codebook_size}.npz")
        if args.build_vocab or (not os.path.exists(vocab_path)):
            print(f"Building vocabulary K={args.codebook_size} (this may take a while)...")
            vocab = build_bovw_vocabulary(
                image_paths=img_files,
                k=args.codebook_size,
                max_total_desc=args.max_total_desc,
                desc_per_image=args.desc_per_image,
                detector=args.detector,
                verbose=True,
            )
            np.savez(vocab_path, centers=vocab)
            print(f"Saved vocabulary to {vocab_path}")
        else:
            data = np.load(vocab_path)
            vocab = data["centers"].astype(np.float32)
            print(f"Loaded vocabulary from {vocab_path} (K={vocab.shape[0]})")

    # Iterate and compute descriptors
    for idx, path in enumerate(img_files):
        name = os.path.basename(path)
        base = os.path.splitext(name)[0]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"[WARN] Failed to read {path}")
            continue
        img_rgb01 = to_rgb01(img_bgr)

        if args.descriptor == "global_color_hist":
            F = extract_global_color_histogram(
                img_rgb01,
                bins_per_channel=args.bins,
                normalize=args.norm,
            )
            params = dict(type="global_color_hist", bins=args.bins, colorspace="rgb", norm=args.norm)
            base = f"{base}_rgb{args.bins}"
        elif args.descriptor == "spatial_grid":
            gr, gc = map(int, args.grid.lower().split("x"))
            F = extract_spatial_grid_color_texture(
                img_rgb01,
                grid=(gr, gc),
                color_bins=args.bins,
                colorspace=args.colorspace,
                lbp_points=args.lbp_points,
                lbp_radius=args.lbp_radius,
                texture=args.texture,
                orient_bins=args.orient_bins,
                unsigned_gradient=(not args.signed),
                color_weight=args.color_weight,
                texture_weight=args.texture_weight,
                color_only=args.color_only,
                texture_only=args.texture_only,
                per_cell_normalize=args.per_cell_normalize,
                final_normalize=args.final_norm,
            )
            params = dict(
                type="spatial_grid",
                grid=f"{gr}x{gc}",
                color_bins=args.bins,
                colorspace=args.colorspace,
                texture=args.texture,
                orient_bins=args.orient_bins,
                unsigned=not args.signed,
                lbp_points=args.lbp_points,
                lbp_radius=args.lbp_radius,
                color_weight=args.color_weight,
                texture_weight=args.texture_weight,
                color_only=args.color_only,
                texture_only=args.texture_only,
                per_cell_normalize=args.per_cell_normalize,
                final_norm=args.final_norm,
            )
            # filename suffix for spatial grid
            if args.texture == 'grad':
                suffix = f"grid{gr}x{gc}_c{args.bins}_grad{args.orient_bins}{'s' if args.signed else 'u'}"
            elif args.texture == 'lbp':
                suffix = f"grid{gr}x{gc}_c{args.bins}_lbp{args.lbp_points}r{args.lbp_radius}"
            else:
                suffix = f"grid{gr}x{gc}_c{args.bins}_colorOnly"
            base = f"{base}_{suffix}"
        else:  # bow
            assert vocab is not None
            F = extract_bovw(
                img=img_bgr,  # BOW uses grayscale; BGR uint8 is fine
                vocab=vocab,
                detector=args.detector,
                max_features=args.max_features,
                normalize=args.norm,
            )
            params = dict(
                type="bow",
                codebook_size=args.codebook_size,
                vocab_file=vocab_path,
                detector=args.detector,
                max_features=args.max_features,
                norm=args.norm,
            )
            base = f"{base}_k{args.codebook_size}_{args.detector}"

        out_path = os.path.join(out_dir, base + ".mat")
        sio.savemat(out_path, {"F": F, "file": name, "params": params})
        if (idx + 1) % 50 == 0 or (idx + 1) == len(img_files):
            print(f"Processed {idx+1}/{len(img_files)} images")

    # Optionally run PCA on the freshly computed folder
    if args.pca_dim is not None or args.pca_var is not None:
        try:
            print("Applying PCA to computed descriptors...")
            # Import local PCA helpers
            from pca import _load_descriptors as _pca_load, _fit_pca as _pca_fit
            X, files = _pca_load(out_dir)
            Z, mu, comps, ev_d, ev_ratio, d, whiten = _pca_fit(
                X,
                dim=args.pca_dim,
                var=args.pca_var,
                whiten=args.pca_whiten,
            )
            out_sub = f"{subfolder}_pca{d}"
            pca_dir = os.path.join(args.out_folder, out_sub)
            os.makedirs(pca_dir, exist_ok=True)
            # Save model
            np.savez(
                os.path.join(pca_dir, "pca_model.npz"),
                mean=mu,
                components=comps,
                explained_variance=ev_d,
                explained_variance_ratio=ev_ratio,
                dim=d,
                whiten=whiten,
                source=subfolder,
            )
            # Save transformed descriptors
            for i, fname in enumerate(files):
                base = os.path.splitext(os.path.basename(fname))[0]
                fout = os.path.join(pca_dir, base + ".mat")
                params = dict(type="pca", dim=d, whiten=whiten, source=subfolder)
                sio.savemat(fout, {"F": Z[i:i+1, :], "file": fname, "params": params})
            # Save a reproducible command hint
            try:
                script_rel = os.path.relpath(os.path.join(os.path.dirname(__file__), 'pca.py'), start=os.getcwd())
                cmd = f"python {script_rel} --desc-subfolder {subfolder} --dim {d}" + (" --whiten" if whiten else "")
                with open(os.path.join(pca_dir, 'command.txt'), 'w') as f:
                    f.write(cmd + "\n")
            except Exception:
                pass
            print(f"Saved PCA descriptors to {pca_dir} (dim={d}, whiten={whiten})")
        except Exception as e:
            print(f"[WARN] PCA step failed: {e}")


if __name__ == "__main__":
    main()
