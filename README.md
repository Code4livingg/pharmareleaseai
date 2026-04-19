# AI-Based Predictive Modelling of Drug Release from Pharmaceutical Formulations Using Machine Learning

This project is a research-grade pharmaceutical engineering workflow for predicting drug release (%) from formulation descriptors such as particle size, loading, encapsulation efficiency, polymer-related variables, process conditions, and time. It includes automated data cleaning, exploratory analysis, model benchmarking, hyperparameter tuning, explainable AI, a Streamlit prediction interface, and report-ready outputs.

## Project Structure

```text
PharmaReleaseAI/
├── .python-version
├── .streamlit/
├── app.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── run_pipeline.py
├── vercel.json
├── requirements.txt
└── src/pharma_release_ai/
```

## How to Use

1. Place `mp_dataset_processed.xlsx` in `data/raw/`.
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Run the full analysis pipeline:

```bash
python3 run_pipeline.py
```

4. Launch the Streamlit interface:

```bash
streamlit run app.py
```

## Deployment

### Vercel

Vercel is not a natural host for Streamlit because Streamlit expects a long-running Python server, while Vercel's Python runtime is request-driven. The repo has been trimmed and productionized so it is safe to import into Vercel, but a native production Streamlit deployment on Vercel is not guaranteed.

1. Push this repository to GitHub.
2. Import the repository into Vercel.
3. Vercel will read `vercel.json` and install the runtime dependencies.

If you need reliable production hosting for this exact Streamlit app, use Streamlit Community Cloud, Render, Railway, or another host that supports persistent Python processes.

### Recommended Deployment For This App

1. Push this repository to GitHub.
2. Import the repository into Streamlit Community Cloud.
3. Select `app.py` as the entrypoint and deploy.

## Outputs Generated

- Cleaned dataset in `data/processed/clean_dataset.csv`
- Trained best model in `models/best_model.pkl`
- Figures in `reports/figures/`
- Model comparison in `reports/model_comparison.csv`
- Explainability table in `reports/feature_importance.csv`
- Full analytical summary in `reports/analysis_summary.json`
- Presentation and viva support material in `reports/*.md`

## Analytical Scope

- Automatic Excel sheet selection
- Target inference with preference for release-related variables
- Missing-value handling and duplicate removal
- Publication-quality EDA plots
- Benchmarking of Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, MLP, and XGBoost when available
- Hyperparameter tuning using cross-validation
- Feature importance, SHAP summary, and partial dependence analysis
- Pharma intelligence statements grounded in data and model evidence

## Notes

- If `xgboost` or `shap` are unavailable, the pipeline still runs with the remaining models and explainability outputs.
- The project is designed to adapt to imperfect column naming and mixed data types.
