from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig, build_default_config
from .data import dataset_summary, load_and_prepare_dataset
from .eda import generate_eda_outputs
from .explainability import generate_explainability_outputs
from .modeling import build_and_train_models
from .reporting import (
    build_future_scope,
    build_ppt_bullets,
    build_report_markdown,
    build_viva_questions,
    write_text,
)
from .utils import save_dataframe, save_joblib, save_json


def run_pipeline(root_dir: str | Path = ".", data_file: str | Path | None = None) -> dict[str, Any]:
    config = build_default_config(root_dir)
    if data_file is not None:
        config.data_file = Path(data_file).resolve()

    for path in [config.output_dir, config.figures_dir, config.models_dir, config.processed_data_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    dataset_bundle = load_and_prepare_dataset(config)
    summary = dataset_summary(dataset_bundle)
    eda_outputs = generate_eda_outputs(dataset_bundle, config)
    modelling_bundle = build_and_train_models(dataset_bundle, config)
    explainability_outputs = generate_explainability_outputs(dataset_bundle, modelling_bundle, config)
    pharma_insights = generate_pharma_intelligence(summary, eda_outputs, explainability_outputs)

    modelling_outputs = {
        "model_results": modelling_bundle.model_results.to_dict(orient="records"),
        "best_model_name": modelling_bundle.best_model_name,
        "tuned_params": modelling_bundle.tuned_params,
        "tuned_metrics": modelling_bundle.tuned_metrics,
        "prediction_interval": modelling_bundle.prediction_interval,
        "feature_columns": modelling_bundle.feature_columns,
        "pharma_insights": pharma_insights,
    }

    save_dataframe(dataset_bundle.clean_df, config.processed_data_path)
    save_dataframe(modelling_bundle.model_results, config.output_dir / "model_comparison.csv")
    save_dataframe(pd.DataFrame(explainability_outputs["importance_table"]), config.output_dir / "feature_importance.csv")
    save_joblib(modelling_bundle.tuned_model, config.models_dir / "best_model.pkl")

    artifacts = {
        "dataset_summary": summary,
        "eda_outputs": eda_outputs,
        "modelling_outputs": modelling_outputs,
        "explainability_outputs": explainability_outputs,
    }
    save_json(artifacts, config.output_dir / "analysis_summary.json")

    report = build_report_markdown(summary, eda_outputs, modelling_outputs, explainability_outputs)
    ppt = build_ppt_bullets(modelling_outputs, summary)
    viva = build_viva_questions()
    future_scope = build_future_scope()

    write_text(config.output_dir / "final_report_summary.md", report)
    write_text(config.output_dir / "ppt_ready_bullets.md", ppt)
    write_text(config.output_dir / "viva_questions_answers.md", viva)
    write_text(config.output_dir / "conclusion_future_scope.md", future_scope)

    return artifacts


def generate_pharma_intelligence(
    dataset_summary: dict[str, Any],
    eda_outputs: dict[str, Any],
    explainability_outputs: dict[str, Any],
) -> list[str]:
    insights: list[str] = []
    correlations = eda_outputs.get("feature_correlations", {})
    importance = explainability_outputs.get("importance_table", [])

    particle_entry = next((item for item in importance if "particle" in item["feature"].lower() or "size" in item["feature"].lower()), None)
    if particle_entry:
        corr = correlations.get(particle_entry["feature"])
        if corr is not None:
            trend = "increase" if corr > 0 else "decrease"
            insights.append(f"Increasing {particle_entry['feature']} tends to {trend} observed drug release in the current dataset.")

    loading_entry = next((item for item in importance if "load" in item["feature"].lower()), None)
    if loading_entry:
        corr = correlations.get(loading_entry["feature"])
        if corr is not None:
            trend = "higher release" if corr > 0 else "lower release"
            insights.append(f"Higher loading correlates with {trend}, subject to the formulation range represented in the source data.")

    encapsulation_entry = next((item for item in importance if "encap" in item["feature"].lower()), None)
    if encapsulation_entry:
        corr = correlations.get(encapsulation_entry["feature"])
        if corr is not None:
            trend = "improved retention and slower release" if corr < 0 else "greater measured release"
            insights.append(f"Encapsulation efficiency is associated with {trend} in this modelling workflow.")

    polymer_entry = next((item for item in importance if "polymer" in item["feature"].lower()), None)
    if polymer_entry:
        insights.append("Polymer-related descriptors materially influence release, indicating matrix composition contributes to sustained-release performance.")

    if not insights:
        top_feature = importance[0]["feature"] if importance else dataset_summary["target_column"]
        insights.append(f"{top_feature} emerged as a major predictor, supporting data-driven formulation optimisation.")

    return insights
