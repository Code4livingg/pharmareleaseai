from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ProjectConfig
from .data import DatasetBundle
from .utils import slugify

sns.set_theme(style="whitegrid", context="talk")


def generate_eda_outputs(bundle: DatasetBundle, config: ProjectConfig) -> dict[str, object]:
    df = bundle.clean_df
    figures_dir = config.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    feature_correlations = (
        numeric_df.corr(numeric_only=True)[bundle.target_column]
        .drop(index=bundle.target_column, errors="ignore")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    histogram_paths = plot_histograms(numeric_df, figures_dir)
    heatmap_path = plot_correlation_heatmap(numeric_df, figures_dir)
    pairplot_path = plot_pairplot(df, bundle, feature_correlations, config, figures_dir)
    scatter_paths = plot_feature_relationships(df, bundle, feature_correlations, figures_dir)
    boxplot_path = plot_outlier_boxplots(numeric_df, figures_dir)
    grouped_trend_path = plot_grouped_release_trend(df, bundle, figures_dir)

    insights = build_eda_insights(bundle, feature_correlations)

    return {
        "feature_correlations": feature_correlations.to_dict(),
        "histogram_paths": histogram_paths,
        "heatmap_path": heatmap_path,
        "pairplot_path": pairplot_path,
        "scatter_paths": scatter_paths,
        "boxplot_path": boxplot_path,
        "grouped_trend_path": grouped_trend_path,
        "insights": insights,
    }


def plot_histograms(numeric_df: pd.DataFrame, figures_dir: Path) -> list[str]:
    paths: list[str] = []
    for column in numeric_df.columns[:8]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(numeric_df[column], kde=True, ax=ax, color="#0f766e")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        path = figures_dir / f"hist_{slugify(column)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_correlation_heatmap(numeric_df: pd.DataFrame, figures_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(numeric_only=True), cmap="crest", annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    path = figures_dir / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return str(path)


def plot_pairplot(
    df: pd.DataFrame,
    bundle: DatasetBundle,
    feature_correlations: pd.Series,
    config: ProjectConfig,
    figures_dir: Path,
) -> str | None:
    top_features = feature_correlations.head(config.max_pairplot_features).index.tolist()
    plot_columns = [col for col in top_features if col in df.columns] + [bundle.target_column]
    plot_columns = list(dict.fromkeys(plot_columns))
    if len(plot_columns) < 3:
        return None
    plot_df = df[plot_columns].dropna()
    if len(plot_df) > 750:
        plot_df = plot_df.sample(n=750, random_state=config.random_seed)
    pairplot = sns.pairplot(plot_df, diag_kind="hist", corner=True, plot_kws={"alpha": 0.6, "s": 28})
    pairplot.fig.suptitle("Pairplot of Top Features", y=1.02)
    path = figures_dir / "pairplot_top_features.png"
    pairplot.savefig(path, dpi=250)
    plt.close("all")
    return str(path)


def plot_feature_relationships(
    df: pd.DataFrame,
    bundle: DatasetBundle,
    feature_correlations: pd.Series,
    figures_dir: Path,
) -> dict[str, str]:
    target = bundle.target_column
    candidates = list(feature_correlations.index)
    labelled_paths: dict[str, str] = {}

    if bundle.time_column:
        labelled_paths["time_vs_release"] = _scatter_with_regression(
            df, bundle.time_column, target, figures_dir / "time_vs_release.png", "Time vs Drug Release"
        )

    keyword_groups = {
        "particle_size_vs_release": ("particle", "size"),
        "loading_vs_release": ("load",),
        "encapsulation_vs_release": ("encapsulation", "encap", "efficiency"),
        "polymer_vs_release": ("polymer",),
    }

    for label, keywords in keyword_groups.items():
        column = next((col for col in candidates if any(word in col.lower() for word in keywords)), None)
        if column:
            labelled_paths[label] = _scatter_with_regression(
                df,
                column,
                target,
                figures_dir / f"{label}.png",
                f"{column} vs {target}",
            )

    return labelled_paths


def _scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter_kws={"alpha": 0.75}, line_kws={"color": "#b91c1c"})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return str(path)


def plot_outlier_boxplots(numeric_df: pd.DataFrame, figures_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=numeric_df, orient="h", ax=ax, color="#7c3aed")
    ax.set_title("Outlier Screening Across Numeric Variables")
    path = figures_dir / "outlier_boxplots.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return str(path)


def plot_grouped_release_trend(df: pd.DataFrame, bundle: DatasetBundle, figures_dir: Path) -> str | None:
    target = bundle.target_column
    grouping_column = next((col for col in bundle.categorical_columns if col not in bundle.id_columns), None)
    if not grouping_column or not bundle.time_column:
        return None
    summary = (
        df.groupby([grouping_column, bundle.time_column], dropna=False)[target]
        .mean()
        .reset_index()
        .sort_values(bundle.time_column)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=summary, x=bundle.time_column, y=target, hue=grouping_column, marker="o", ax=ax)
    ax.set_title("Release Trend Grouped by Formulation")
    path = figures_dir / "release_trend_grouped_by_formulation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return str(path)


def build_eda_insights(bundle: DatasetBundle, feature_correlations: pd.Series) -> list[str]:
    insights: list[str] = []
    if not feature_correlations.empty:
        strongest = feature_correlations.head(3)
        for feature, corr in strongest.items():
            direction = "positive" if corr > 0 else "negative"
            insights.append(
                f"{feature} shows a {direction} association with {bundle.target_column} (Pearson r={corr:.3f})."
            )

    if bundle.time_column and bundle.time_column in feature_correlations.index:
        corr = feature_correlations[bundle.time_column]
        insights.append(
            f"Time appears to be a major driver of release kinetics, with correlation magnitude {abs(corr):.3f}."
        )

    particle_feature = next(
        (col for col in feature_correlations.index if "particle" in col.lower() or "size" in col.lower()),
        None,
    )
    if particle_feature:
        corr = feature_correlations[particle_feature]
        verb = "higher" if corr > 0 else "lower"
        insights.append(
            f"Particle-size-related behaviour suggests larger values are linked to {verb} observed release in this dataset."
        )

    polymer_feature = next((col for col in feature_correlations.index if "polymer" in col.lower()), None)
    if polymer_feature:
        insights.append(
            f"Polymer-related descriptors appear influential enough to warrant mechanistic interpretation during formulation optimisation."
        )

    return insights
