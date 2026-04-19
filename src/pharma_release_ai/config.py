from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProjectConfig:
    root_dir: Path
    data_file: Path
    output_dir: Path
    figures_dir: Path
    models_dir: Path
    processed_data_path: Path
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    max_pairplot_features: int = 5
    top_pdp_features: int = 3
    preferred_target_keywords: tuple[str, ...] = (
        "release",
        "drug release",
        "release percentage",
        "release %",
        "% release",
        "cumulative release",
    )
    ignored_identifier_keywords: tuple[str, ...] = (
        "id",
        "identifier",
        "sample",
        "batch",
        "run",
        "index",
        "no.",
        "sr.",
    )
    likely_time_keywords: tuple[str, ...] = ("time", "hour", "hr", "min", "minute", "day")
    confidence_interval_alpha: float = 0.05
    sheet_name: Optional[str] = None


def build_default_config(root_dir: str | Path) -> ProjectConfig:
    root = Path(root_dir).resolve()
    return ProjectConfig(
        root_dir=root,
        data_file=root / "data" / "raw" / "mp_dataset_processed.xlsx",
        output_dir=root / "reports",
        figures_dir=root / "reports" / "figures",
        models_dir=root / "models",
        processed_data_path=root / "data" / "processed" / "clean_dataset.csv",
    )
