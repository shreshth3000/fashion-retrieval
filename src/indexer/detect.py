"""Garment detection: isolate individual garments so downstream stages reason per-garment
instead of per-image. This per-garment split is what makes compositional binding (query 5,
"red tie AND white shirt") possible at all."""

from dataclasses import dataclass

from PIL import Image

from src.common.config import load_config
from src.common.models import get_detector


@dataclass
class Detection:
    crop: Image.Image
    bbox: tuple[float, float, float, float]
    category: str
    score: float


def detect_garments(image: Image.Image) -> list[Detection]:
    detector = get_detector()
    threshold = load_config()["models"]["detector_score_threshold"]
    raw = detector(image)

    detections = []
    for det in raw:
        if det["score"] < threshold:
            continue
        box = det["box"]
        bbox = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
        crop = image.crop(bbox)
        detections.append(
            Detection(crop=crop, bbox=bbox, category=det["label"], score=det["score"])
        )
    return detections
