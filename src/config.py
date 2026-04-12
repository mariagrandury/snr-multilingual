import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_tasks(key: str) -> list[str]:
    with open(CONFIGS_DIR / "tasks.json") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"Task key '{key}' not found. Available: {list(data.keys())}")
    return data[key]


def load_models(key: str) -> list[dict]:
    with open(CONFIGS_DIR / "models.json") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"Model key '{key}' not found. Available: {list(data.keys())}")
    return data[key]
