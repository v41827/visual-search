import numpy as np
import cv2
from typing import Iterable, List, Optional, Sequence, Tuple


def extractRandom(img: np.ndarray) -> np.ndarray:
    """
    Legacy placeholder that returns a 1x30 random descriptor.
    Expects a normalized RGB image (float in [0,1]).
    """
    F = np.random.rand(1, 30)
    return F


def extract_random(img: np.ndarray) -> np.ndarray:
    """Snake_case alias for extractRandom."""
    return extractRandom(img)


# -----------------------------
# Global Colour Histogram (3D)
# -----------------------------
def _quantize_3d_hist(img: np.ndarray, bins_per_channel: int) -> np.ndarray:
    """
    Compute a 3D colour histogram over an RGB image using equal-width binning per channel.

    - img: normalized RGB float image in [0,1], shape (H,W,3)
    - bins_per_channel: number of bins per channel (e.g., 4, 8, 16, 32, 64)

    Returns a 1D histogram of length bins_per_channel^3 (not normalized).
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Expected RGB image"
    b = bins_per_channel
    # Map [0,1] -> [0, b-1]
    q = np.floor(np.clip(img, 0.0, 0.999999) * b).astype(np.int32)
    # 3D bin index
    idx = q[:, :, 0] * (b * b) + q[:, :, 1] * b + q[:, :, 2]
    hist = np.bincount(idx.ravel(), minlength=b * b * b).astype(np.float64)
    return hist


def extract_global_color_histogram(
    img: np.ndarray,
    bins_per_channel: int = 8,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Global 3D colour histogram.

    - img: normalized RGB float image in [0,1], shape (H,W,3)
    - bins_per_channel: 4, 8, 16, 32, 64
    - normalize: 'l1', 'l2', or 'none'

    Returns a row vector (1, D) where D = bins_per_channel^3.
    """
    # Only RGB colours (as per coursework requirement)
    work = img

    hist = _quantize_3d_hist(work, bins_per_channel)
    if normalize == "l1":
        s = hist.sum() + 1e-12
        hist = hist / s
    elif normalize == "l2":
        n = np.linalg.norm(hist) + 1e-12
        hist = hist / n
    # else 'none'
    return hist.reshape(1, -1)


# -----------------------------
# Texture: Local Binary Patterns
# -----------------------------
def _lbp_hist(
    gray: np.ndarray,
    n_points: int = 8,
    radius: int = 1,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Compute LBP (basic) codes histogram on a grayscale image.
    - gray: float image in [0,1], shape (H,W)
    - n_points: number of neighbors (supports 8 only in this basic implementation)
    - radius: neighbor radius (>=1)
    - normalize: 'l1', 'l2', or 'none'

    Returns a 1D histogram of length 2^n_points.
    """
    assert gray.ndim == 2, "Expected grayscale image"
    if n_points != 8:
        # Basic implementation supports 8 neighbors; silently degrade to 8
        n_points = 8
    R = int(radius)
    H, W = gray.shape
    if H < 2 * R + 1 or W < 2 * R + 1:
        # Too small; return all zeros
        hist = np.zeros(1 << n_points, dtype=np.float64)
        return hist

    def crop_shift(im: np.ndarray, dy: int, dx: int, r: int) -> np.ndarray:
        return im[r + dy : H - r + dy, r + dx : W - r + dx]

    center = crop_shift(gray, 0, 0, R)
    neighbors = [
        crop_shift(gray, -R, -R, R),
        crop_shift(gray, -R, 0, R),
        crop_shift(gray, -R, +R, R),
        crop_shift(gray, 0, +R, R),
        crop_shift(gray, +R, +R, R),
        crop_shift(gray, +R, 0, R),
        crop_shift(gray, +R, -R, R),
        crop_shift(gray, 0, -R, R),
    ]

    codes = np.zeros_like(center, dtype=np.uint16)
    for bit, nb in enumerate(neighbors):
        codes |= ((nb >= center).astype(np.uint16) << bit)

    hist = np.bincount(codes.ravel(), minlength=1 << n_points).astype(np.float64)
    if normalize == "l1":
        s = hist.sum() + 1e-12
        hist = hist / s
    elif normalize == "l2":
        n = np.linalg.norm(hist) + 1e-12
        hist = hist / n
    return hist


# -----------------------------
# Texture: Gradient Orientation
# -----------------------------
def _grad_orient_hist(
    gray: np.ndarray,
    bins: int = 16,
    unsigned: bool = True,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Magnitude-weighted gradient orientation histogram.
    - gray: float [0,1] grayscale, shape (H,W)
    - bins: number of orientation bins (e.g., 8,16,32)
    - unsigned: True -> angles in [0,180), False -> [0,360)
    - normalize: 'l1', 'l2', or 'none'
    """
    assert gray.ndim == 2, "Expected grayscale image"
    # Compute Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.degrees(np.arctan2(gy, gx))  # [-180,180]
    if unsigned:
        # Map to [0,180)
        ang = np.mod(ang, 180.0)
        period = 180.0
    else:
        # Map to [0,360)
        ang = np.mod(ang + 360.0, 360.0)
        period = 360.0

    # Bin indices
    bin_w = period / float(bins)
    idx = np.floor(ang / bin_w).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)

    hist = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=bins).astype(np.float64)
    if normalize == "l1":
        s = hist.sum() + 1e-12
        hist = hist / s
    elif normalize == "l2":
        n = np.linalg.norm(hist) + 1e-12
        hist = hist / n
    return hist

# -----------------------------------------
# Spatial Grid (Colour and Texture via LBP)
# -----------------------------------------
def extract_spatial_grid_color_texture(
    img: np.ndarray,
    grid: Tuple[int, int] = (2, 2),
    color_bins: int = 8,
    colorspace: str = "rgb",
    lbp_points: int = 8,
    lbp_radius: int = 1,
    texture: str = "lbp",
    orient_bins: int = 16,
    unsigned_gradient: bool = True,
    color_weight: float = 1.0,
    texture_weight: float = 1.0,
    color_only: bool = False,
    texture_only: bool = False,
    per_cell_normalize: bool = True,
    final_normalize: str = "l2",
) -> np.ndarray:
    """
    Compute a spatial grid descriptor concatenating per-cell 3D colour histograms and LBP histograms.

    - img: normalized RGB float image in [0,1], shape (H,W,3)
    - grid: (rows, cols)
    - color_bins: number of colour bins per channel (3D hist => color_bins^3 per cell)
    - colorspace: 'rgb' or 'hsv' for colour histogram
    - texture: 'none', 'lbp', or 'grad'
    - lbp_points, lbp_radius: texture parameters when texture='lbp'
    - orient_bins, unsigned_gradient: parameters when texture='grad'
    - color_weight, texture_weight: weights multiplying color/texture per-cell histograms
    - color_only / texture_only: boolean switches to include only one branch
    - per_cell_normalize: L1-normalize each sub-histogram before concatenation
    - final_normalize: 'l1', 'l2', or 'none' applied to the concatenated vector

    Returns a row vector (1, D) concatenating selected per-cell histograms.
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Expected RGB image"
    H, W, _ = img.shape
    gr, gc = grid
    # Precompute colour-space representation
    if colorspace.lower() == "hsv":
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 0] /= 179.0
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0
        work_color = hsv
    else:
        work_color = img

    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Determine cell boundaries
    rows = np.linspace(0, H, gr + 1, dtype=int)
    cols = np.linspace(0, W, gc + 1, dtype=int)

    include_color = (not texture_only) and (color_weight > 0)
    include_texture = (not color_only) and (texture.lower() != "none") and (texture_weight > 0)

    feats: List[np.ndarray] = []
    for i in range(gr):
        for j in range(gc):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            cell_color = work_color[r0:r1, c0:c1, :]
            cell_gray = gray[r0:r1, c0:c1]

            if include_color:
                ch = _quantize_3d_hist(cell_color, color_bins)
                if per_cell_normalize:
                    ch = ch / (ch.sum() + 1e-12)
                if color_weight != 1.0:
                    ch = ch * float(color_weight)
                feats.append(ch)

            if include_texture:
                tmode = texture.lower()
                if tmode == "lbp":
                    th = _lbp_hist(cell_gray, n_points=lbp_points, radius=lbp_radius, normalize="none")
                elif tmode == "grad":
                    th = _grad_orient_hist(cell_gray, bins=orient_bins, unsigned=unsigned_gradient, normalize="none")
                else:
                    th = None
                if th is not None:
                    if per_cell_normalize:
                        th = th / (th.sum() + 1e-12)
                    if texture_weight != 1.0:
                        th = th * float(texture_weight)
                    feats.append(th)

    F = np.concatenate(feats, axis=0)
    if final_normalize == "l1":
        F = F / (F.sum() + 1e-12)
    elif final_normalize == "l2":
        F = F / (np.linalg.norm(F) + 1e-12)
    return F.reshape(1, -1)


# -----------------------------
# Bag of Visual Words (SIFT)
# -----------------------------
def _sift_detector(max_features: Optional[int] = None):
    try:
        if max_features is None:
            return cv2.SIFT_create()
        else:
            return cv2.SIFT_create(nfeatures=int(max_features))
    except Exception as e:
        raise RuntimeError(
            "OpenCV SIFT not available. Ensure you're using a build with xfeatures or SIFT enabled."
        ) from e


def _harris_keypoints(
    gray_u8: np.ndarray,
    max_features: int = 500,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
) -> List[cv2.KeyPoint]:
    """
    Detect Harris corners and return as OpenCV KeyPoints (no orientation).
    - gray_u8: uint8 grayscale
    - max_features: cap number of returned points (strongest first)
    """
    gray_f32 = gray_u8.astype(np.float32) / 255.0
    R = cv2.cornerHarris(gray_f32, block_size, ksize, k)
    # Non-maximum suppression by dilation
    R_dil = cv2.dilate(R, None)
    mask = (R == R_dil)
    # Flatten and sort by response
    ys, xs = np.where(mask)
    if ys.size == 0:
        return []
    responses = R[ys, xs]
    order = np.argsort(responses)[::-1]
    xs = xs[order]
    ys = ys[order]
    kps: List[cv2.KeyPoint] = []
    for x, y in zip(xs[:max_features], ys[:max_features]):
        # OpenCV Python expects positional args: (x, y, size)
        kps.append(cv2.KeyPoint(float(x), float(y), 3))
    return kps


def _detect_and_describe(
    gray_u8: np.ndarray,
    method: str = "sift",
    max_features: int = 500,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Detect keypoints using 'sift' or 'harris', and compute SIFT descriptors.
    Returns (keypoints, descriptors or None).
    """
    method = method.lower()
    sift = _sift_detector(max_features)
    if method == "sift":
        return sift.detectAndCompute(gray_u8, None)
    elif method == "harris":
        kps = _harris_keypoints(gray_u8, max_features=max_features)
        if not kps:
            return [], None
        kps, des = sift.compute(gray_u8, kps)
        return kps, des
    else:
        # default to sift
        return sift.detectAndCompute(gray_u8, None)


def build_bovw_vocabulary(
    image_paths: Sequence[str],
    k: int = 200,
    max_total_desc: int = 100000,
    desc_per_image: int = 500,
    detector: str = "sift",
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a BoVW vocabulary using SIFT and k-means (OpenCV implementation).

    - image_paths: list of image file paths
    - k: codebook size (#clusters)
    - max_total_desc: cap on total SIFT descriptors used for clustering
    - desc_per_image: cap of descriptors per image

    Returns centers with shape (k, 128) float32.
    """
    all_desc: List[np.ndarray] = []
    total = 0
    for idx, p in enumerate(image_paths):
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = _detect_and_describe(gray, method=detector, max_features=desc_per_image)
        if des is None or des.size == 0:
            continue
        if des.shape[0] > desc_per_image:
            des = des[:desc_per_image]
        all_desc.append(des)
        total += des.shape[0]
        if verbose and (idx % 100 == 0):
            print(f"Collected {total} descriptors from {idx+1} images")
        if total >= max_total_desc:
            break

    if not all_desc:
        raise RuntimeError("No SIFT descriptors collected to build vocabulary")

    data = np.vstack(all_desc).astype(np.float32)
    if data.shape[0] > max_total_desc:
        data = data[:max_total_desc]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    flags = cv2.KMEANS_PP_CENTERS
    attempts = 3
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, flags)
    return centers.astype(np.float32)


def extract_bovw(
    img: np.ndarray,
    vocab: np.ndarray,
    detector: str = "sift",
    max_features: int = 500,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Compute a BoVW histogram for a given image using a provided vocabulary.

    - img: uint8 BGR image or normalized RGB; only grayscale is used internally
    - vocab: (k, 128) float32 array of SIFT cluster centers
    - max_features: SIFT nfeatures cap used in detection
    - normalize: 'l1', 'l2', or 'none'

    Returns a row vector (1, k)
    """
    # Ensure grayscale uint8
    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img

    kps, des = _detect_and_describe(gray, method=detector, max_features=max_features)
    k = vocab.shape[0]
    hist = np.zeros(k, dtype=np.float64)
    if des is None or des.size == 0:
        # no features -> empty histogram
        pass
    else:
        des = des.astype(np.float32)
        # Assign each descriptor to nearest center
        # Compute squared distances efficiently: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        a2 = np.sum(des * des, axis=1, keepdims=True)  # (N,1)
        b2 = np.sum(vocab * vocab, axis=1)[None, :]    # (1,K)
        ab = des @ vocab.T                              # (N,K)
        d2 = a2 + b2 - 2.0 * ab
        assign = np.argmin(d2, axis=1)
        hist = np.bincount(assign, minlength=k).astype(np.float64)

    if normalize == "l1":
        hist = hist / (hist.sum() + 1e-12)
    elif normalize == "l2":
        hist = hist / (np.linalg.norm(hist) + 1e-12)
    return hist.reshape(1, -1)
