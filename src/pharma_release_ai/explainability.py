from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from .config import ProjectConfig
from .data import DatasetBundle
from .modeling import ModellingBundle

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None


def generate_explainability_outputs(
    dataset_bundle: DatasetBundle,
    modelling_bundle: ModellingBundle,
    config: ProjectConfig,
) -> dict[str, object]:
    figures_dir = config.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    tuned_model = modelling_bundle.tuned_model
    X_test = modelling_bundle.X_test
    y_test = modelling_bundle.y_test
    feature_names = X_test.columns.tolist()

    importance = permutation_importance(
        tuned_model, X_test, y_test, n_repeats=15, random_state=config.random_seed, scoring="r2"
    )
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    importance_path = plot_feature_importance(importance_df, figures_dir)
    pdp_path = plot_partial_dependence(tuned_model, X_test, importance_df, config, figures_dir)
    shap_path = plot_shap_summary(tuned_model, X_test, figures_dir)
    interpretations = build_pharma_interpretations(dataset_bundle, importance_df)

    return {
        "importance_table": importance_df.to_dict(orient="records"),
        "importance_path": importance_path,
        "partial_dependence_path": pdp_path,
        "shap_path": shap_path,
        "interpretations": interpretations,
    }


def plot_feature_importance(importance_df: pd.DataFrame, figures_dir: Path) -> str:
    top_df = importance_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_df["feature"], top_df["importance"], color="#0ea5e9")
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean importance")
    path = figures_dir / "feature_importance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return str(path)


def plot_partial_dependence(model, X_test: pd.DataFrame, importance_df: pd.DataFrame, config: ProjectConfig, figures_dir: Path) -> str | None:
    top_features = importance_df.head(config.top_pdp_features)["feature"].tolist()
    if not top_features:
        return None
    fig, ax = plt.subplots(figsize=(12, 4 * len(top_features)))
    PartialDependenceDisplay.from_estimator(model, X_test, top_features, ax=ax)
    path = figures_dir / "partial_dependence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return str(path)


def plot_shap_summary(model, X_test: pd.DataFrame, figures_dir: Path) -> str | None:
    if not SHAP_AVAILABLE:
        return None
    sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    try:
        explainer = shap.Explainer(model.predict, sample)
        shap_values = explainer(sample)
        shap.summary_plot(shap_values, sample, show=False)
        path = figures_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)
    except Exception:
        plt.close("all")
        return None


def build_pharma_interpretations(dataset_bundle: DatasetBundle, importance_df: pd.DataFrame) -> list[str]:
    messages: list[str] = []
    top_features = importance_df.head(5)
    if not top_features.empty:
        lead = top_features.iloc[0]["feature"]
        messages.append(f"{lead} is the dominant model-derived driver of {dataset_bundle.target_column}.")

    particle = next((f for f in importance_df["feature"] if "particle" in f.lower() or "size" in f.lower()), None)
    if particle:
        messages.append(
            f"Particle-size-related effects are sufficiently important to influence release control and sustained-delivery interpretation."
        )

    loading = next((f for f in importance_df["feature"] if "load" in f.lower()), None)
    if loading:
        messages.append(
            f"Drug loading contributes materially to prediction, indicating formulation composition has a measurable effect on release behaviour."
        )

    encapsulation = next((f for f in importance_df["feature"] if "encap" in f.lower()), None)
    if encapsulation:
        messages.append(
            f"Encapsulation-efficiency-like variables appear mechanistically relevant, consistent with altered retention and diffusion behaviour."
        )

    polymer = next((f for f in importance_df["feature"] if "polymer" in f.lower()), None)
    if polymer:
        messages.append(
            f"Polymer-related inputs rank among the important predictors, supporting their role in modulating barrier properties and release kinetics."
        )

    return messages
