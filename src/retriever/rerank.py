"""Stage 3 — precision. A dual-encoder (FashionCLIP) embeds image and text independently, so
composition is lost at encode time. BLIP-ITM ingests image and text jointly and attends across
both, so it can resolve binding (e.g. "red tie AND white shirt") that Stage 1 cannot. Runs only
on the Stage-2 survivors, which keeps it affordable regardless of DB size."""

import torch
from PIL import Image

from src.common.config import load_config
from src.common.models import get_reranker


def rerank_score(image_path: str, query_text: str) -> float:
    model, processor = get_reranker()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=query_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.itm_score.softmax(dim=-1)[0, 1].item()


def rerank_candidates(
    candidates: list[dict], query_text: str, weights: dict[str, float] | None = None
) -> list[dict]:
    for c in candidates:
        c["rerank_score"] = rerank_score(c["image_path"], query_text)

    weights = weights or load_config()["retrieval"]["weights"]
    for c in candidates:
        c["final_score"] = (
            weights["stage1_similarity"] * c["stage1_similarity"]
            + weights["attribute_match"] * c["attribute_match"]
            + weights["scene_match"] * c["scene_match"]
            + weights["rerank_score"] * c["rerank_score"]
        )

    return sorted(candidates, key=lambda c: c["final_score"], reverse=True)
