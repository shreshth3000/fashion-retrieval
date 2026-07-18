"""Whole-image environment classification, scored independently from garments so "where"
stays decoupled from "what" — this is why filter.py can boost scene matches without
conflating them with garment-attribute matches."""

import torch
from PIL import Image

from src.common.config import load_config
from src.common.models import get_fashion_clip


def classify_scene(image: Image.Image) -> tuple[str, dict[str, float]]:
    cfg = load_config()["scene"]
    labels = cfg["labels"]
    prompts = [cfg["prompt_template"].format(label=label) for label in labels]

    model, processor = get_fashion_clip()
    inputs = processor(text=prompts, images=[image], return_tensors="pt", padding=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0).cpu().tolist()

    scores = dict(zip(labels, probs))
    best_label = max(scores, key=scores.get)
    return best_label, scores
