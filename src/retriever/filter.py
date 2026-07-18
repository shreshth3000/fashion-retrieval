"""Stage 2 — gate + boost. Hard-drop candidates missing a required (garment, color) pair;
boost candidates whose scene/formality match. Cheap, and removes obvious false positives
before the expensive reranker runs."""

import re

from src.common.schema import Query
from src.indexer.color import color_matches


def _type_matches(detector_label: str, garment_type: str) -> bool:
    """Fuzzy match between the detector's raw category label (e.g. Fashionpedia's
    "shirt, blouse") and the parser's controlled vocabulary term (e.g. "shirt"). An exact
    equality check is too brittle — detector taxonomies rarely line up 1:1 with a hand-picked
    query vocabulary (also handles cases like query "raincoat" vs. detector "coat")."""
    label_words = [w for w in re.split(r"[,\s]+", detector_label.lower()) if w]
    gt_words = [w for w in re.split(r"[,\s]+", garment_type.lower()) if w]
    return any(gw in lw or lw in gw for gw in gt_words for lw in label_words)


def _garment_satisfied(garments: list[dict], garment_type: str, color_label: str) -> bool:
    for g in garments:
        if not _type_matches(g["type"], garment_type):
            continue
        if not color_label:
            return True
        if color_matches(tuple(g["color_lab"]), color_label):
            return True
    return False


def apply_filter(candidates: list[dict], query: Query) -> list[dict]:
    survivors = []
    for c in candidates:
        garments = c["garments"]

        dropped = False
        for garment_type, color_label in query.garment_color_pairs:
            if not _garment_satisfied(garments, garment_type, color_label):
                dropped = True
                break
        if dropped:
            continue

        attribute_match = 1.0 if query.garment_color_pairs else 0.5

        scene_score = 1.0 if query.scene and c["scene_label"] == query.scene else 0.0
        formality_score = (
            1.0
            if query.formality and any(g["formality"] == query.formality for g in garments)
            else 0.0
        )
        active_axes = int(bool(query.scene)) + int(bool(query.formality))
        scene_match = (scene_score + formality_score) / active_axes if active_axes else 0.5

        c["attribute_match"] = attribute_match
        c["scene_match"] = scene_match
        survivors.append(c)

    return survivors
