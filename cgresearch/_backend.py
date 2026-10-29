"""Make the prototype's flat-import backend modules importable without moving them.
Looks for <repo>/backend first (flat layout), then <repo>/../backend (nested research/ layout)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
BACKEND = next((p for p in (_ROOT / "backend", _ROOT.parent / "backend") if p.is_dir()), _ROOT / "backend")
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
