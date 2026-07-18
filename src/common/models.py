"""Loads and caches every model used by the indexer + retriever, so each one loads once
per process regardless of how many stages need it."""

import functools

import torch
from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

from src.common.config import load_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def get_fashion_clip() -> tuple[CLIPModel, CLIPProcessor]:
    model_id = load_config()["models"]["fashion_clip"]
    model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


@functools.lru_cache(maxsize=1)
def get_detector():
    cfg = load_config()["models"]
    return pipeline(
        "object-detection",
        model=cfg["detector"],
        device=0 if DEVICE == "cuda" else -1,
    )


@functools.lru_cache(maxsize=1)
def get_reranker() -> tuple[BlipForImageTextRetrieval, BlipProcessor]:
    model_id = load_config()["models"]["reranker"]
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(DEVICE).eval()
    processor = BlipProcessor.from_pretrained(model_id)
    return model, processor
