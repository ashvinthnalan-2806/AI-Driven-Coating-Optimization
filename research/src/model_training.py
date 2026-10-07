"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Module: src/model_training.py
Description: Multi-output regression model training, benchmarking, automatic
             model selection, and joblib persistence.
=============================================================================
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Construct candidate multi-output regression models for benchmarking.

    Includes:
    1. Random Forest Regressor (Native multi-output)
    2. Extra Trees Regressor (Extremely randomized trees, native multi-output)
    3. Gradient Boosting Regressor (Wrapped via MultiOutputRegressor)
    4. Ridge Regression (Regularized linear baseline wrapped via MultiOutputRegressor)
    """
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                random_state=random_state
            )
        ),
        "Ridge Linear Baseline": MultiOutputRegressor(
            Ridge(alpha=1.0)
        )
    }
    return models


def evaluate_model_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    target_names: list
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Calculate R2, MAE, and RMSE for each target property individually,
    as well as macro-averaged performance across all targets.

    Parameters
    ----------
    model : Any
        Trained multi-output regression model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.DataFrame
        True test target properties.
    target_names : list
        List of target property names.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], Dict[str, float]]
        (per_target_metrics, overall_metrics)
    """
    y_pred = model.predict(X_test)
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    per_target = {}
    r2_list, mae_list, rmse_list = [], [], []

    for i, target in enumerate(target_names):
        true_col = y_test[target].values
        pred_col = y_pred[:, i]

        r2 = float(r2_score(true_col, pred_col))
        mae = float(mean_absolute_error(true_col, pred_col))
        rmse = float(root_mean_squared_error(true_col, pred_col))

        per_target[target] = {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        }

        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)

    overall = {
        "Mean_R2": float(np.mean(r2_list)),
        "Mean_MAE": float(np.mean(mae_list)),
        "Mean_RMSE": float(np.mean(rmse_list))
    }

    return per_target, overall


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    target_names: list,
    random_state: int = 42
) -> Tuple[str, Any, pd.DataFrame, Dict[str, Any]]:
    """
    Train all candidate models, evaluate each on test predictions, and select
    the best performing architecture based on macro-averaged test R2.

    Returns
    -------
    Tuple[str, Any, pd.DataFrame, Dict[str, Any]]
        (best_model_name, best_model, comparison_df, detailed_results)
    """
    candidates = get_candidate_models(random_state=random_state)
    comparison_records = []
    detailed_results = {}

    print(f"Training and benchmarking {len(candidates)} candidate regression models...")

    for name, model in candidates.items():
        print(f"  -> Fitting {name} on {X_train.shape[0]} training samples...")
        model.fit(X_train, y_train)

        per_target, overall = evaluate_model_performance(model, X_test, y_test, target_names)
        detailed_results[name] = {
            "model": model,
            "per_target": per_target,
            "overall": overall
        }

        comparison_records.append({
            "Model": name,
            "Mean R²": overall["Mean_R2"],
            "Mean MAE": overall["Mean_MAE"],
            "Mean RMSE": overall["Mean_RMSE"],
            "Hardness R²": per_target["Hardness"]["R2"],
            "Adhesion R²": per_target["Adhesion"]["R2"],
            "Flexibility R²": per_target["Flexibility"]["R2"],
            "Gloss R²": per_target["Gloss"]["R2"],
            "Drying_Time R²": per_target["Drying_Time"]["R2"],
        })

    comparison_df = pd.DataFrame(comparison_records).sort_values(by="Mean R²", ascending=False).reset_index(drop=True)

    best_name = comparison_df.iloc[0]["Model"]
    best_model = detailed_results[best_name]["model"]

    print(f"\n[Selection] Top Performing Model: '{best_name}' (Mean R²: {comparison_df.iloc[0]['Mean R²']:.4f})")

    return best_name, best_model, comparison_df, detailed_results


def save_model_bundle(
    model: Any,
    model_name: str,
    feature_names: list,
    target_names: list,
    metrics: Dict[str, Any],
    filepath: str
) -> None:
    """
    Persist trained model, metadata, feature names, and test metrics using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "feature_names": feature_names,
        "target_names": target_names,
        "metrics": metrics,
        "created_at": pd.Timestamp.now().isoformat()
    }
    joblib.dump(bundle, filepath)
    print(f"[Model Saved] Model bundle saved to: {filepath}")


def load_model_bundle(filepath: str) -> Dict[str, Any]:
    """
    Load persisted model bundle from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model bundle not found at: {filepath}")
    bundle = joblib.load(filepath)
    return bundle
