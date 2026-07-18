"""FashionCLIP embeddings. Whole-image embedding is the Stage-1 ANN recall unit; per-crop
embeddings are stored on each garment record for optional finer-grained matching."""

import torch
from PIL import Image

from src.common.models import get_fashion_clip


def _embed_images(images: list[Image.Image]) -> list[list[float]]:
    # Bypass model.get_image_features(): on this checkpoint it returns the raw
    # BaseModelOutputWithPooling instead of the projected tensor on some transformers
    # versions. Call the vision tower + projection directly instead (same path
    # get_image_features uses internally).
    model, processor = get_fashion_clip()
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        pooled_output = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        features = model.visual_projection(pooled_output)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().tolist()


def embed_image(image: Image.Image) -> list[float]:
    return _embed_images([image])[0]


def embed_text(text: str) -> list[float]:
    model, processor = get_fashion_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        pooled_output = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        ).pooler_output
        features = model.text_projection(pooled_output)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().tolist()[0]
