"""Read existing scene splits without reassigning scenes."""
from pathlib import Path
import json


def validate_split(payload):
    result = {}
    seen = set()
    for split in ("train", "val", "test"):
        rows = payload.get(split)
        if not isinstance(rows, list) or not all(isinstance(x, str) and x for x in rows):
            raise ValueError(f"{split} must be a list of scene IDs")
        if len(rows) != len(set(rows)) or seen.intersection(rows):
            raise ValueError(f"duplicate/overlapping scene IDs in {split}")
        result[split] = list(rows)
        seen.update(rows)
    return result


def load_split(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
    else:
        import yaml
        payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("split manifest must be a mapping")
    return validate_split(payload)
