from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .utils import normalise_column_name


@dataclass
class DatasetBundle:
    raw_sheets: dict[str, pd.DataFrame]
    selected_sheet: str
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    target_column: str
    time_column: str | None
    id_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_report: dict[str, int]
    duplicate_rows: int
    outlier_report: dict[str, int]
    variable_descriptions: dict[str, str]
    sheet_scores: dict[str, float]


def load_and_prepare_dataset(config: ProjectConfig) -> DatasetBundle:
    if not config.data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.data_file}. Place mp_dataset_processed.xlsx in data/raw/."
        )

    excel = pd.ExcelFile(config.data_file)
    raw_sheets = {sheet: excel.parse(sheet) for sheet in excel.sheet_names}
    scores = {sheet: score_sheet(df) for sheet, df in raw_sheets.items()}
    selected_sheet = config.sheet_name or max(scores, key=scores.get)
    raw_df = raw_sheets[selected_sheet].copy()
    raw_df.columns = [normalise_column_name(col) for col in raw_df.columns]

    clean_df = raw_df.copy()
    clean_df = clean_df.dropna(axis=1, how="all")
    clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()]
    clean_df = clean_df.replace(r"^\s*$", np.nan, regex=True)

    for column in clean_df.columns:
        if clean_df[column].dtype == object:
            converted = pd.to_numeric(clean_df[column], errors="ignore")
            clean_df[column] = converted

    duplicate_rows = int(clean_df.duplicated().sum())
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    missing_report = clean_df.isna().sum().to_dict()
    numeric_columns = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = clean_df.select_dtypes(exclude=[np.number]).columns.tolist()

    for column in numeric_columns:
        clean_df[column] = clean_df[column].fillna(clean_df[column].median())
    for column in categorical_columns:
        mode = clean_df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        clean_df[column] = clean_df[column].fillna(fill_value)

    target_column = detect_target_column(clean_df, config)
    time_column = detect_time_column(clean_df, config, exclude={target_column})
    id_columns = detect_identifier_columns(clean_df, config, exclude={target_column})
    outlier_report = detect_outliers(clean_df, exclude=[target_column])
    variable_descriptions = build_variable_descriptions(clean_df, target_column, time_column, id_columns)

    return DatasetBundle(
        raw_sheets=raw_sheets,
        selected_sheet=selected_sheet,
        raw_df=raw_df,
        clean_df=clean_df,
        target_column=target_column,
        time_column=time_column,
        id_columns=id_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_report={k: int(v) for k, v in missing_report.items()},
        duplicate_rows=duplicate_rows,
        outlier_report=outlier_report,
        variable_descriptions=variable_descriptions,
        sheet_scores=scores,
    )


def score_sheet(df: pd.DataFrame) -> float:
    if df.empty:
        return float("-inf")
    non_empty_ratio = 1.0 - df.isna().all(axis=1).mean()
    numeric_ratio = (df.apply(pd.api.types.is_numeric_dtype)).mean()
    release_bonus = sum("release" in str(col).lower() for col in df.columns)
    return float(df.shape[0] * non_empty_ratio + numeric_ratio * 10 + release_bonus * 25)


def detect_target_column(df: pd.DataFrame, config: ProjectConfig) -> str:
    lower_map = {column: column.lower() for column in df.columns}
    for keyword in config.preferred_target_keywords:
        candidates = [column for column, lowered in lower_map.items() if keyword in lowered]
        if candidates:
            numeric_candidates = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
            return numeric_candidates[0] if numeric_candidates else candidates[0]

    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_candidates:
        raise ValueError("No numeric columns available to infer a target variable.")
    variability = {col: df[col].nunique(dropna=True) for col in numeric_candidates}
    return max(variability, key=variability.get)


def detect_time_column(df: pd.DataFrame, config: ProjectConfig, exclude: set[str]) -> str | None:
    for column in df.columns:
        lowered = column.lower()
        if column in exclude:
            continue
        if any(keyword in lowered for keyword in config.likely_time_keywords):
            return column
    return None


def detect_identifier_columns(df: pd.DataFrame, config: ProjectConfig, exclude: set[str]) -> list[str]:
    identifiers: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if column in exclude:
            continue
        if any(keyword == lowered or keyword in lowered for keyword in config.ignored_identifier_keywords):
            identifiers.append(column)
            continue
        if df[column].nunique(dropna=True) == len(df) and not pd.api.types.is_numeric_dtype(df[column]):
            identifiers.append(column)
    return identifiers


def detect_outliers(df: pd.DataFrame, exclude: list[str] | None = None) -> dict[str, int]:
    exclude = set(exclude or [])
    outliers: dict[str, int] = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        if column in exclude:
            continue
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers[column] = 0
            continue
        mask = (df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)
        outliers[column] = int(mask.sum())
    return outliers


def build_variable_descriptions(
    df: pd.DataFrame,
    target_column: str,
    time_column: str | None,
    id_columns: list[str],
) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for column in df.columns:
        if column == target_column:
            descriptions[column] = "Primary response variable representing drug release behaviour."
        elif time_column and column == time_column:
            descriptions[column] = "Time-related process variable influencing release kinetics."
        elif column in id_columns:
            descriptions[column] = "Identifier-like metadata column excluded from predictive modelling."
        elif pd.api.types.is_numeric_dtype(df[column]):
            descriptions[column] = "Numeric formulation or process descriptor used for analysis and modelling."
        else:
            descriptions[column] = "Categorical formulation descriptor encoded for machine learning."
    return descriptions


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    df = bundle.clean_df
    return {
        "selected_sheet": bundle.selected_sheet,
        "sheet_scores": bundle.sheet_scores,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": bundle.target_column,
        "time_column": bundle.time_column,
        "numeric_columns": bundle.numeric_columns,
        "categorical_columns": bundle.categorical_columns,
        "identifier_columns": bundle.id_columns,
        "missing_report": bundle.missing_report,
        "duplicate_rows_removed": bundle.duplicate_rows,
        "outlier_report": bundle.outlier_report,
        "variable_descriptions": bundle.variable_descriptions,
    }
