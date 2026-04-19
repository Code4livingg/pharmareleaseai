from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .config import ProjectConfig
from .data import DatasetBundle

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None


@dataclass
class ModellingBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocess: ColumnTransformer
    model_results: pd.DataFrame
    trained_models: dict[str, Pipeline]
    best_model_name: str
    best_model: Pipeline
    tuned_model: Pipeline
    tuned_params: dict[str, Any]
    tuned_metrics: dict[str, float]
    prediction_interval: float
    feature_columns: list[str]


def build_and_train_models(bundle: DatasetBundle, config: ProjectConfig) -> ModellingBundle:
    df = bundle.clean_df.copy()
    target = bundle.target_column
    feature_columns = [col for col in df.columns if col != target and col not in bundle.id_columns]
    X = df[feature_columns]
    y = df[target]

    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_seed
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=config.random_seed),
        "Lasso": Lasso(random_state=config.random_seed),
        "Random Forest": RandomForestRegressor(random_state=config.random_seed, n_estimators=300),
        "Gradient Boosting": GradientBoostingRegressor(random_state=config.random_seed),
        "SVR": SVR(),
        "MLP Regressor": MLPRegressor(random_state=config.random_seed, max_iter=1500),
    }
    if XGBOOST_AVAILABLE and XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            random_state=config.random_seed,
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
        )

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)

    results: list[dict[str, float | str]] = []
    trained_models: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append(
            {
                "Model": name,
                "R2": r2_score(y_test, predictions),
                "RMSE": compute_rmse(y_test, predictions),
                "MAE": mean_absolute_error(y_test, predictions),
                "CV_Mean_R2": float(np.mean(cv_results["test_r2"])),
                "CV_Mean_RMSE": float(-np.mean(cv_results["test_rmse"])),
                "CV_Mean_MAE": float(-np.mean(cv_results["test_mae"])),
            }
        )
        trained_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by=["R2", "CV_Mean_R2"], ascending=False).reset_index(drop=True)
    best_model_name = str(results_df.iloc[0]["Model"])
    best_model = trained_models[best_model_name]
    tuned_model, tuned_params, tuned_metrics = tune_best_model(
        best_model_name, preprocess, X_train, y_train, X_test, y_test, config
    )
    residuals = y_test - tuned_model.predict(X_test)
    prediction_interval = float(np.quantile(np.abs(residuals), 0.95)) if len(residuals) else 0.0

    return ModellingBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocess=preprocess,
        model_results=results_df,
        trained_models=trained_models,
        best_model_name=best_model_name,
        best_model=best_model,
        tuned_model=tuned_model,
        tuned_params=tuned_params,
        tuned_metrics=tuned_metrics,
        prediction_interval=prediction_interval,
        feature_columns=feature_columns,
    )


def tune_best_model(
    best_model_name: str,
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: ProjectConfig,
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    estimator, param_grid = tuning_space(best_model_name, config)
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed),
        n_jobs=None,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    predictions = best.predict(X_test)
    metrics = {
        "R2": float(r2_score(y_test, predictions)),
        "RMSE": float(compute_rmse(y_test, predictions)),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "CV_Best_R2": float(search.best_score_),
    }
    return best, search.best_params_, metrics


def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def tuning_space(best_model_name: str, config: ProjectConfig) -> tuple[Any, dict[str, list[Any]]]:
    if best_model_name == "Linear Regression":
        return LinearRegression(), {}
    if best_model_name == "Ridge":
        return Ridge(random_state=config.random_seed), {"model__alpha": [0.1, 1.0, 10.0, 50.0]}
    if best_model_name == "Lasso":
        return Lasso(random_state=config.random_seed), {"model__alpha": [0.001, 0.01, 0.1, 1.0]}
    if best_model_name == "Random Forest":
        return RandomForestRegressor(random_state=config.random_seed), {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 6, 12],
            "model__min_samples_split": [2, 5],
        }
    if best_model_name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=config.random_seed), {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        }
    if best_model_name == "SVR":
        return SVR(), {"model__C": [1, 10, 50], "model__gamma": ["scale", "auto"], "model__kernel": ["rbf"]}
    if best_model_name == "MLP Regressor":
        return MLPRegressor(random_state=config.random_seed, max_iter=2000), {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__alpha": [0.0001, 0.001, 0.01],
        }
    if best_model_name == "XGBoost" and XGBOOST_AVAILABLE and XGBRegressor is not None:
        return XGBRegressor(random_state=config.random_seed, objective="reg:squarederror"), {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        }
    return LinearRegression(), {}
