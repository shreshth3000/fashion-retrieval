"""Single point of entry for config.yaml. Nothing else should open the file directly."""

import functools
from pathlib import Path

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
load_dotenv(CONFIG_PATH.parent / ".env")


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolves a config.yaml path against the repo root, not the process cwd — cwd differs
    between running a script from the repo root and running a notebook from notebooks/."""
    return CONFIG_PATH.parent / relative_path
