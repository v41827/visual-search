import numpy as np


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[0] == 1:
        return a.ravel()
    if a.ndim > 1:
        return a.reshape(-1)
    return a


def l2_distance(F1: np.ndarray, F2: np.ndarray) -> float:
    x = _as_1d(F1)
    y = _as_1d(F2)
    return float(np.linalg.norm(x - y))


def l1_distance(F1: np.ndarray, F2: np.ndarray) -> float:
    x = _as_1d(F1)
    y = _as_1d(F2)
    return float(np.sum(np.abs(x - y)))


def chi_square_distance(F1: np.ndarray, F2: np.ndarray, eps: float = 1e-12) -> float:
    x = _as_1d(F1)
    y = _as_1d(F2)
    num = (x - y) ** 2
    den = x + y + eps
    return float(0.5 * np.sum(num / den))


def histogram_intersection(F1: np.ndarray, F2: np.ndarray) -> float:
    x = _as_1d(F1)
    y = _as_1d(F2)
    # Return a distance (smaller is better), 1 - intersection
    inter = float(np.sum(np.minimum(x, y)))
    return 1.0 - inter

def mahalanobis_distance(F1: np.ndarray, F2: np.ndarray, VI: np.ndarray) -> float: #i just added this sec - codex please check for me if the implementation/ theory corerct or not
    x = _as_1d(F1)
    y = _as_1d(F2)
    diff = x - y
    dist = np.sqrt(np.dot(np.dot(diff.T, VI), diff))
    return float(dist)

def cvpr_compare(F1: np.ndarray, F2: np.ndarray) -> float:
    """
    Default comparison: Euclidean (L2) distance, as required for the
    baseline global colour histogram.
    """
    return l2_distance(F1, F2)
