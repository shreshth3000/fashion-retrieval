"""Orchestrates Part A: for every image in data/images/, run detection -> per-garment
color/formality -> whole-image embedding + scene classification, then write everything into
the Chroma index (embedding + flattened metadata)."""

import argparse
import json
from collections import Counter

from PIL import Image
from tqdm import tqdm

from src.common.config import load_config, resolve_path
from src.indexer.attributes import classify_formality
from src.indexer.color import dominant_color
from src.indexer.detect import detect_garments
from src.indexer.embed import embed_image
from src.indexer.scene import classify_scene
from src.store.vector_store import get_collection, upsert_image


def build_image_record(image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")

    detections = detect_garments(image)
    garments = []
    for det in detections:
        color_label, color_lab = dominant_color(det.crop)
        formality = classify_formality(det.crop)
        garments.append(
            {
                "type": det.category,
                "color_label": color_label,
                "color_lab": color_lab,
                "formality": formality,
                "bbox": det.bbox,
            }
        )

    scene_label, scene_scores = classify_scene(image)
    image_embedding = embed_image(image)

    return {
        "image_embedding": image_embedding,
        "scene_label": scene_label,
        "scene_scores": scene_scores,
        "garments": garments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap number of images indexed")
    args = parser.parse_args()

    cfg = load_config()["paths"]
    images_dir = resolve_path(cfg["images_dir"])
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if args.limit:
        image_paths = image_paths[: args.limit]

    collection = get_collection()
    scene_counter = Counter()

    for image_path in tqdm(image_paths, desc="indexing"):
        image_id = image_path.stem
        record = build_image_record(image_path)
        scene_counter[record["scene_label"]] += 1
        upsert_image(
            collection,
            image_id=image_id,
            embedding=record["image_embedding"],
            metadata={
                "scene_label": record["scene_label"],
                "scene_scores": record["scene_scores"],
                "garments": record["garments"],
            },
            image_path=str(image_path),
        )

    print(f"indexed {len(image_paths)} images")
    print("scene distribution:", json.dumps(scene_counter, indent=2))


if __name__ == "__main__":
    main()
