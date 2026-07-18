import os

from src.retriever.parse_query import _fallback_parse, parse_query


def test_fallback_extracts_garment_and_color():
    q = _fallback_parse("A person in a bright yellow raincoat.")
    assert ("raincoat", "yellow") in q.garment_color_pairs


def test_fallback_extracts_tie_and_shirt():
    q = _fallback_parse("A red tie and a white shirt in a formal setting.")
    garments = {g for g, _ in q.garment_color_pairs}
    assert "tie" in garments
    assert "shirt" in garments
    assert q.formality == "formal"


def test_fallback_scene_inference():
    q = _fallback_parse("Someone wearing a blue shirt sitting on a park bench.")
    assert q.scene == "park"


def test_parse_query_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    q = parse_query("Professional business attire inside a modern office.")
    assert q.raw_text.startswith("Professional")
