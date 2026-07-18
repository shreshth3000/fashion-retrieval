"""Run the 5 assignment eval queries end-to-end and save top-k image grids as PNGs.
No notebook required.

Usage: python scripts/eval.py
Requires the index already built (python -m src.indexer.build_index).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless — no GUI, no notebook, just writes files
import matplotlib.pyplot as plt
from PIL import Image

from src.common.config import resolve_path
from src.retriever.query import run_query

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

OUTPUT_DIR = resolve_path("eval_output")


def save_grid(query_text: str, results: list[dict], out_path: Path) -> None:
    if not results:
        print(f"  no results for: {query_text}")
        return
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    fig.suptitle(query_text)
    for ax, r in zip(axes, results):
        ax.imshow(Image.open(r["image_path"]))
        ax.set_title(f"score={r['final_score']:.3f}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, query_text in enumerate(EVAL_QUERIES, start=1):
        start = time.time()
        print(f"[{i}/{len(EVAL_QUERIES)}] {query_text}")
        results = run_query(query_text)
        for rank, r in enumerate(results, start=1):
            print(f"    {rank}. {r['image_path']}  score={r['final_score']:.4f}")

        out_path = OUTPUT_DIR / f"query_{i}.png"
        save_grid(query_text, results, out_path)
        print(f"    saved grid -> {out_path}  ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
