"""Dataclasses shared by the indexer and retriever."""

from dataclasses import dataclass, field


@dataclass
class GarmentRecord:
    type: str
    color_label: str
    color_lab: tuple[float, float, float]
    formality: str
    bbox: tuple[float, float, float, float]
    crop_embedding: list[float] | None = None


@dataclass
class ImageRecord:
    image_id: str
    image_path: str
    image_embedding: list[float]
    scene_label: str
    scene_scores: dict[str, float]
    garments: list[GarmentRecord] = field(default_factory=list)


@dataclass
class Query:
    raw_text: str
    garment_color_pairs: list[tuple[str, str]] = field(default_factory=list)
    scene: str | None = None
    formality: str | None = None
