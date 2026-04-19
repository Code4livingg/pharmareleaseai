from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_report_markdown(
    dataset_summary: dict[str, Any],
    eda_outputs: dict[str, Any],
    modelling_outputs: dict[str, Any],
    explainability_outputs: dict[str, Any],
) -> str:
    table = pd.DataFrame(modelling_outputs["model_results"]).to_markdown(index=False)
    insights = "\n".join(f"- {item}" for item in eda_outputs["insights"])
    xai = "\n".join(f"- {item}" for item in explainability_outputs["interpretations"])
    pharma_layer = "\n".join(f"- {item}" for item in modelling_outputs["pharma_insights"])
    return f"""# Final Report Summary

## Dataset Summary
- Selected sheet: {dataset_summary["selected_sheet"]}
- Rows: {dataset_summary["rows"]}
- Columns: {dataset_summary["columns"]}
- Target variable: {dataset_summary["target_column"]}
- Time variable: {dataset_summary["time_column"]}
- Duplicate rows removed: {dataset_summary["duplicate_rows_removed"]}

## Missing Value Report
{pd.Series(dataset_summary["missing_report"]).to_markdown()}

## Key EDA Insights
{insights}

## Model Comparison
{table}

## Best Tuned Model
- Best baseline model: {modelling_outputs["best_model_name"]}
- Tuned holdout R2: {modelling_outputs["tuned_metrics"]["R2"]:.4f}
- Tuned holdout RMSE: {modelling_outputs["tuned_metrics"]["RMSE"]:.4f}
- Tuned holdout MAE: {modelling_outputs["tuned_metrics"]["MAE"]:.4f}
- Best CV score during tuning: {modelling_outputs["tuned_metrics"]["CV_Best_R2"]:.4f}

## Explainability
{xai}

## Pharma Intelligence Layer
{pharma_layer}

## Conclusion
The tuned {modelling_outputs["best_model_name"]} workflow demonstrates that machine learning can capture complex, nonlinear relationships in pharmaceutical formulation data and provide a reproducible basis for drug-release optimisation.
"""


def build_ppt_bullets(modelling_outputs: dict[str, Any], dataset_summary: dict[str, Any]) -> str:
    return f"""# PPT-Ready Bullet Points

- Objective: Predict drug release (%) from formulation and process variables using machine learning.
- Dataset used: {dataset_summary["rows"]} rows and {dataset_summary["columns"]} columns from the selected Excel sheet.
- Analytical workflow: cleaning, EDA, feature engineering, multi-model benchmarking, tuning, and explainable AI.
- Best baseline model selected automatically: {modelling_outputs["best_model_name"]}.
- Tuned model performance: R2={modelling_outputs["tuned_metrics"]["R2"]:.3f}, RMSE={modelling_outputs["tuned_metrics"]["RMSE"]:.3f}, MAE={modelling_outputs["tuned_metrics"]["MAE"]:.3f}.
- Feature importance and partial dependence were used to identify the strongest formulation drivers.
- The framework can reduce trial-and-error experimentation in sustained-release formulation design.
- Streamlit deployment enables rapid prediction for new formulation inputs.
"""


def build_viva_questions() -> str:
    return """# Viva Questions and Answers

## 1. Why is machine learning suitable for drug release prediction?
Machine learning is suitable because drug release depends on nonlinear interactions among formulation variables, process parameters, and time, which are difficult to capture using simple empirical equations alone.

## 2. Why were multiple models compared?
Multiple models were compared to determine whether the dataset is better represented by linear, regularised, kernel-based, ensemble, or neural-network approaches.

## 3. What is the significance of R2, RMSE, and MAE?
R2 indicates explained variance, RMSE penalises larger prediction errors, and MAE shows the average absolute deviation between predicted and observed release.

## 4. Why is explainable AI important in pharmaceutical projects?
Explainability is important because formulation decisions must be scientifically interpretable, not only accurate, especially in regulated and research-intensive settings.

## 5. How does hyperparameter tuning improve the model?
Hyperparameter tuning searches for settings that improve generalisation performance and reduce underfitting or overfitting on unseen data.

## 6. What are the limitations of this study?
Limitations include dataset size, possible formulation imbalance, measurement noise, and dependence on the range covered by the source experiments.

## 7. How can this work be extended?
Future work can include external validation, mechanistic-ML hybrid models, larger formulation datasets, and optimisation engines for inverse formulation design.
"""


def build_future_scope() -> str:
    return """# Conclusions and Future Scope

## Conclusion
This project establishes a reproducible machine-learning workflow for predicting drug release from formulation descriptors and process variables. By combining exploratory analysis, model benchmarking, tuning, and explainable AI, it supports data-driven optimisation of pharmaceutical formulations.

## Future Scope
- Add external validation datasets from additional formulation studies.
- Integrate mechanistic release models with data-driven predictors.
- Extend to classification of immediate, controlled, and sustained release profiles.
- Use Bayesian optimisation for inverse design of formulations targeting desired release behaviour.
- Deploy a validated interface for lab-side decision support.
"""


def write_text(path: str | Path, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
