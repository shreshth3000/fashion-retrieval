"""Dominant garment color via LAB clustering. LAB (not RGB/HSV) because Euclidean distance in
LAB space tracks perceptual difference, which keeps "is this yellow enough" thresholding
stable across lighting conditions."""

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans

from src.common.config import load_config

# Reference sRGB (0-255) for each vocabulary color, used to build a LAB lookup table.
_REFERENCE_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (220, 20, 20),
    "orange": (255, 140, 0),
    "yellow": (255, 220, 0),
    "green": (34, 139, 34),
    "blue": (30, 80, 220),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "brown": (101, 67, 33),
    "beige": (222, 202, 172),
    "navy": (0, 0, 90),
    "maroon": (110, 15, 30),
    "olive": (107, 107, 30),
}


def _reference_lab_table() -> dict[str, np.ndarray]:
    vocab = load_config()["color"]["vocabulary"]
    table = {}
    for name in vocab:
        rgb = np.array(_REFERENCE_RGB[name], dtype=np.float64).reshape(1, 1, 3) / 255.0
        table[name] = rgb2lab(rgb).reshape(3)
    return table


def dominant_color(crop: Image.Image) -> tuple[str, tuple[float, float, float]]:
    """Returns (nearest vocabulary label, raw LAB centroid) for the crop's dominant color."""
    cfg = load_config()["color"]
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float64) / 255.0
    lab = rgb2lab(rgb).reshape(-1, 3)

    n_clusters = min(cfg["kmeans_clusters"], len(lab))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(lab)
    counts = np.bincount(kmeans.labels_)
    centroid = kmeans.cluster_centers_[counts.argmax()]

    reference = _reference_lab_table()
    label = min(reference, key=lambda name: np.linalg.norm(reference[name] - centroid))
    return label, tuple(centroid.tolist())


def lab_distance(lab_a: tuple[float, float, float], lab_b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(lab_a) - np.array(lab_b)))


def color_matches(color_lab: tuple[float, float, float], target_label: str) -> bool:
    """Hard-gate check used by Stage 2: is `color_lab` close enough to `target_label`?"""
    cfg = load_config()["color"]
    reference = _reference_lab_table()
    if target_label not in reference:
        return False
    return lab_distance(color_lab, tuple(reference[target_label])) <= cfg["lab_match_threshold"]
