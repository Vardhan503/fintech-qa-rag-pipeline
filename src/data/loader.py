"""Read the raw FinQA dev split."""

from src.utils.io import RAW_DIR, load_json

DEFAULT_RAW_PATH = RAW_DIR / "dev.json"


def load_raw_finqa(path=DEFAULT_RAW_PATH) -> list[dict]:
    """One entry per QA example. Page content is duplicated across every
    question asked about the same page; reconstruction collapses that later."""
    return load_json(path)
