from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pharma_release_ai.pipeline import run_pipeline


if __name__ == "__main__":
    artifacts = run_pipeline(PROJECT_ROOT)
    print("Pipeline completed.")
    print(f"Best model: {artifacts['modelling_outputs']['best_model_name']}")
