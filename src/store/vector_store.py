"""Thin wrapper over ChromaDB. Image embeddings go into the ANN index; garment/scene/
formality metadata rides alongside each entry for Stage 2 filtering."""

import json

import chromadb

from src.common.config import load_config


def get_collection():
    cfg = load_config()["paths"]
    client = chromadb.PersistentClient(path=cfg["chroma_dir"])
    return client.get_or_create_collection(
        name=cfg["chroma_collection"], metadata={"hnsw:space": "cosine"}
    )


def upsert_image(collection, image_id: str, embedding: list[float], metadata: dict, image_path: str) -> None:
    # Chroma metadata values must be scalar; nested structures (garments) are JSON-encoded.
    flat_metadata = {
        "image_path": image_path,
        "scene_label": metadata["scene_label"],
        "scene_scores": json.dumps(metadata["scene_scores"]),
        "garments": json.dumps(metadata["garments"]),
    }
    collection.upsert(ids=[image_id], embeddings=[embedding], metadatas=[flat_metadata])


def query_top_n(collection, query_embedding: list[float], n: int) -> list[dict]:
    result = collection.query(query_embeddings=[query_embedding], n_results=n)
    candidates = []
    for i in range(len(result["ids"][0])):
        meta = result["metadatas"][0][i]
        candidates.append(
            {
                "image_id": result["ids"][0][i],
                "image_path": meta["image_path"],
                "scene_label": meta["scene_label"],
                "scene_scores": json.loads(meta["scene_scores"]),
                "garments": json.loads(meta["garments"]),
                "stage1_distance": result["distances"][0][i],
            }
        )
    return candidates
