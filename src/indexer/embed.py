"""FashionCLIP embeddings. Whole-image embedding is the Stage-1 ANN recall unit; per-crop
embeddings are stored on each garment record for optional finer-grained matching."""

import torch
from PIL import Image

from src.common.models import get_fashion_clip


def _embed_images(images: list[Image.Image]) -> list[list[float]]:
    model, processor = get_fashion_clip()
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().tolist()


def embed_image(image: Image.Image) -> list[float]:
    return _embed_images([image])[0]


def embed_text(text: str) -> list[float]:
    model, processor = get_fashion_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().tolist()[0]
