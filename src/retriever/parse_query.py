"""NL query -> structured constraints. Gemini 2.5 Flash (text-only) does the heavy lifting on
soft phrasing ("casual weekend city walk"); a regex/keyword fallback covers the case where
GEMINI_API_KEY is unset or the API call fails, using the same controlled vocabulary."""

import json
import os
import re

from src.common.config import load_config
from src.common.schema import Query

_JSON_SCHEMA_PROMPT = """You extract structured fashion-retrieval constraints from a natural
language query. Return ONLY valid JSON, no markdown fences, matching this shape:

{{
  "garment_color_pairs": [["<garment>", "<color>"], ...],
  "scene": "<one of {scenes} or null>",
  "formality": "<one of {formalities} or null>"
}}

Rules:
- garment must be one of: {garments}
- color must be one of: {colors}
- Only include a garment_color_pair when the query mentions or strongly implies a garment.
  If a garment is mentioned without a color, still include it with color "" (empty string).
- Infer scene/formality from soft phrasing (e.g. "city walk" -> scene "urban street",
  "weekend" -> formality "casual") even if not stated explicitly.

Query: "{query}"
"""


def _fallback_parse(text: str) -> Query:
    cfg = load_config()
    text_lower = text.lower()

    pairs = []
    for garment in cfg["garment_types"]:
        if re.search(rf"\b{re.escape(garment)}\b", text_lower):
            color = ""
            for c in cfg["color"]["vocabulary"]:
                if re.search(rf"\b{re.escape(c)}\b", text_lower):
                    # crude proximity check: color word within 3 tokens of garment word
                    color = c
                    break
            pairs.append((garment, color))

    scene = next((s for s in cfg["scene"]["labels"] if s.split()[0] in text_lower), None)
    formality = next((f for f in cfg["formality"]["labels"] if f in text_lower), None)

    return Query(raw_text=text, garment_color_pairs=pairs, scene=scene, formality=formality)


def _llm_parse(text: str) -> Query:
    from google import genai

    cfg = load_config()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = _JSON_SCHEMA_PROMPT.format(
        scenes=cfg["scene"]["labels"],
        formalities=cfg["formality"]["labels"],
        garments=cfg["garment_types"],
        colors=cfg["color"]["vocabulary"],
        query=text,
    )
    response = client.models.generate_content(
        model=cfg["models"]["query_parser_llm"],
        contents=prompt,
        config={
            "temperature": cfg["query_parser"]["llm_temperature"],
            "response_mime_type": "application/json",
        },
    )
    data = json.loads(response.text)
    pairs = [(g, c) for g, c in data.get("garment_color_pairs", [])]
    return Query(
        raw_text=text,
        garment_color_pairs=pairs,
        scene=data.get("scene"),
        formality=data.get("formality"),
    )


def parse_query(text: str) -> Query:
    cfg = load_config()["query_parser"]
    if cfg["use_llm"] and os.environ.get("GEMINI_API_KEY"):
        try:
            return _llm_parse(text)
        except Exception:
            pass  # fall through to regex parser
    return _fallback_parse(text)
