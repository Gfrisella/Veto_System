"""Imported first by every script in bin/: puts lib/ on the path, non-interactive plots."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
os.environ.setdefault("MPLBACKEND", "Agg")
