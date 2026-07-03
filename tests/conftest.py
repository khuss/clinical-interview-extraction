"""Make `src/` importable as top-level modules (schema, extract, derive, ...),
matching the convention every src/*.py script already uses (sys.path.insert)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
