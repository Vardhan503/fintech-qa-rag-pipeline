import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_finqa   # noqa: E402
from src.utils.io import RAW_DIR             # noqa: E402


@pytest.fixture(scope="session")
def raw_data() -> list[dict]:
    if not (RAW_DIR / "dev.json").exists():
        pytest.skip("data/raw/dev.json not present")
    return load_raw_finqa()
