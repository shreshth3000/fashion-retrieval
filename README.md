# Fashion Retrieval — Multimodal Fashion & Context Retrieval

Natural-language image retrieval over a fashion dataset. Beats vanilla CLIP on
compositional, multi-attribute queries (garment+color, scene, formality) via a
retrieve -> filter -> rerank pipeline. See `config.yaml` for every model/threshold/weight
used — nothing is hardcoded in the Python modules.

## Architecture

```
Query --> [parse]  --> {(garment,color) pairs, scene, formality}
                          |
Image DB --> Stage 1: FashionCLIP ANN search      --> top ~50 candidates (recall)
             Stage 2: attribute pre-filter + boost --> gated/reweighted set
             Stage 3: BLIP-ITM cross-encoder rerank --> top-k (precision)
```

- **Stage 1 (`src/retriever/search.py`)** — FashionCLIP dual-encoder ANN recall.
- **Stage 2 (`src/retriever/filter.py`)** — hard-drops candidates missing a required
  (garment, color) pair using LAB color distance; boosts scene/formality matches.
- **Stage 3 (`src/retriever/rerank.py`)** — BLIP-ITM cross-encoder jointly attends over
  image and full query text, resolving compositional binding
  ("red tie AND white shirt") that the Stage-1 dual-encoder cannot.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Query parsing uses Gemini 2.5 Flash (text-only) with a regex fallback. Copy `.env.example` to
`.env` and fill in your key (or `export` it directly) — falls back to regex parsing if unset:

```bash
cp .env.example .env   # then edit .env
```

## Build the index (Part A)

```bash
python scripts/fetch_dataset.py          # samples ~1800 images from Fashionpedia into data/
python -m src.indexer.build_index        # detect -> color -> embed -> scene -> Chroma
```

`build_index.py` prints the scene-label distribution at the end — Fashionpedia skews toward
street-style photography, so check that office/home aren't badly underrepresented.

## Query (Part B)

```bash
python -m src.retriever.query "a bright yellow raincoat" --k 5
```

Or run `notebooks/eval.ipynb` to execute all 5 assignment queries and view top-k image grids.

## Tests

```bash
pytest tests/
```

## Repo layout

```
src/indexer/     Part A — feature extraction, writes to the vector store
src/retriever/   Part B — NL query -> top-k images
src/store/       thin ChromaDB wrapper
src/common/      shared config/model-loading/schema
config.yaml      every model name, path, k, and score weight
```

See `DELIVERABLE.md` for the approaches write-up, shortcomings, and future work.
