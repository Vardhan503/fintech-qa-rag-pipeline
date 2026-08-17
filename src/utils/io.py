"""JSONL read/write plus project-root path resolution.

Relative paths are resolved against the repository root rather than the current
working directory, so the same path string works from a notebook in notebooks/,
a script in scripts/, or a pytest run at the root.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path) -> list[dict]:
    with open(resolve(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path) -> Path:
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return target


def load_json(path):
    with open(resolve(path)) as f:
        return json.load(f)


def save_json(obj, path, indent: int = 2) -> Path:
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(obj, f, indent=indent)
    return target
