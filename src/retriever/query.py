"""CLI entrypoint: text in -> top-k image paths + scores out. All logic lives in the stage
modules; this file just wires them together.

Usage: python -m src.retriever.query "a bright yellow raincoat" --k 5
"""

import argparse

from src.common.config import load_config
from src.retriever.filter import apply_filter
from src.retriever.parse_query import parse_query
from src.retriever.rerank import rerank_candidates
from src.retriever.search import recall_candidates


def run_query(text: str, k: int | None = None) -> list[dict]:
    k = k or load_config()["retrieval"]["final_k"]

    query = parse_query(text)
    candidates = recall_candidates(text)
    survivors = apply_filter(candidates, query)

    weights = None
    if not survivors:
        # Hard gate wiped the field (e.g. detector missed the garment) — fall back to
        # Stage 1 recall alone rather than returning nothing. attribute_match/scene_match
        # are structurally 0 here (not "no match", just "not evaluated"), so redistribute
        # their weight onto the two signals that are still meaningful instead of letting
        # them silently deflate every fallback-path score.
        survivors = candidates
        for c in survivors:
            c["attribute_match"] = 0.0
            c["scene_match"] = 0.0
        base_weights = load_config()["retrieval"]["weights"]
        live_total = base_weights["stage1_similarity"] + base_weights["rerank_score"]
        weights = {
            "stage1_similarity": base_weights["stage1_similarity"] / live_total,
            "attribute_match": 0.0,
            "scene_match": 0.0,
            "rerank_score": base_weights["rerank_score"] / live_total,
        }

    ranked = rerank_candidates(survivors, text, weights=weights)
    return ranked[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    results = run_query(args.query, args.k)
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. {r['image_path']}  score={r['final_score']:.4f}")


if __name__ == "__main__":
    main()
