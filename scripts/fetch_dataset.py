"""One-time data acquisition: sample images from the Fashionpedia HF dataset into
data/images/, keep original bbox/category annotations in data/metadata.jsonl as ground truth
for sanity-checking detect.py later. Not part of the ML pipeline — run once before indexing.

Usage: python scripts/fetch_dataset.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.common.config import load_config


def main() -> None:
    cfg = load_config()
    ds_cfg = cfg["dataset"]
    paths_cfg = cfg["paths"]

    images_dir = Path(paths_cfg["images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(paths_cfg["metadata_jsonl"])

    rng = random.Random(ds_cfg["seed"])
    target = ds_cfg["target_num_images"]

    pooled = []
    for split in ds_cfg["splits"]:
        split_ds = load_dataset(ds_cfg["hf_dataset_id"], split=split)
        pooled.extend((split, i) for i in range(len(split_ds)))
    rng.shuffle(pooled)
    pooled = pooled[:target]

    loaded_splits = {}
    written = 0
    with open(metadata_path, "w") as meta_file:
        for split, idx in pooled:
            if split not in loaded_splits:
                loaded_splits[split] = load_dataset(ds_cfg["hf_dataset_id"], split=split)
            example = loaded_splits[split][idx]

            image_id = f"{split}_{example['image_id']}"
            image_path = images_dir / f"{image_id}.jpg"
            example["image"].convert("RGB").save(image_path)

            meta_file.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "width": example["width"],
                        "height": example["height"],
                        "objects": example["objects"],
                    }
                )
                + "\n"
            )
            written += 1

    print(f"wrote {written} images to {images_dir}, metadata at {metadata_path}")


if __name__ == "__main__":
    main()
