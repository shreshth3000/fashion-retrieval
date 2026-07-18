"""Per-garment attributes: type comes straight from the detector's category label; formality
is zero-shot classified from the garment crop via FashionCLIP text prompts."""

import torch
from PIL import Image

from src.common.config import load_config
from src.common.models import get_fashion_clip


def classify_formality(crop: Image.Image) -> str:
    cfg = load_config()["formality"]
    labels = cfg["labels"]
    prompts = [cfg["prompt_template"].format(label=label) for label in labels]

    model, processor = get_fashion_clip()
    inputs = processor(text=prompts, images=[crop], return_tensors="pt", padding=True).to(
        model.device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0).cpu().tolist()
    return labels[probs.index(max(probs))]
