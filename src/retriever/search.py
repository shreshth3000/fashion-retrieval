"""Stage 1 — recall. Embed the full query with FashionCLIP, ANN search Chroma, return the
top-n. This stage is biased for recall; Stage 2/3 clean up precision."""

from src.common.config import load_config
from src.indexer.embed import embed_text
from src.store.vector_store import get_collection, query_top_n


def recall_candidates(query_text: str) -> list[dict]:
    top_n = load_config()["retrieval"]["stage1_top_n"]
    query_embedding = embed_text(query_text)
    collection = get_collection()
    candidates = query_top_n(collection, query_embedding, top_n)
    for c in candidates:
        c["stage1_similarity"] = 1.0 - c["stage1_distance"]  # cosine distance -> similarity
    return candidates
