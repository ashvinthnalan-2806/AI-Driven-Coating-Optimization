"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Module: src/evaluation.py
Description: Rigorous model evaluation, residual analysis, actual vs. predicted
             diagnostics, and permutation feature importance analysis.
=============================================================================
"""

import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


sns.set_theme(style="whitegrid", font="sans-serif")


def compute_property_metrics_table(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_names: List[str]
) -> pd.DataFrame:
    """
    Format property-by-property evaluation metrics into a publication table.
    """
    records = []
    for i, tgt in enumerate(target_names):
        true_vals = y_true[tgt].values
        pred_vals = y_pred[:, i]
        records.append({
            "Property": tgt,
            "R²": round(float(r2_score(true_vals, pred_vals)), 4),
            "MAE": round(float(mean_absolute_error(true_vals, pred_vals)), 3),
            "RMSE": round(float(root_mean_squared_error(true_vals, pred_vals)), 3)
        })
    metrics_df = pd.DataFrame(records)
    return metrics_df


def plot_actual_vs_predicted(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Actual vs. Predicted scatter plots with y = x ideal parity lines.
    """
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(4.2 * n_targets, 4.2))

    for i, tgt in enumerate(target_names):
        ax = axes[i]
        true_vals = y_true[tgt].values
        pred_vals = y_pred[:, i]

        r2 = r2_score(true_vals, pred_vals)
        rmse = root_mean_squared_error(true_vals, pred_vals)

        # Scatter points
        ax.scatter(true_vals, pred_vals, alpha=0.6, color="#1f77b4", edgecolor="black", s=40)

        # Ideal parity line (y = x)
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        padding = (max_val - min_val) * 0.05
        line_range = np.linspace(min_val - padding, max_val + padding, 100)
        ax.plot(line_range, line_range, color="#d62728", linestyle="--", linewidth=1.8, label="Ideal (y=x)")

        ax.set_title(f"{tgt}\nR²: {r2:.3f} | RMSE: {rmse:.2f}", fontweight="bold", fontsize=11)
        ax.set_xlabel(f"Actual {tgt}")
        ax.set_ylabel(f"Predicted {tgt}")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)

    fig.suptitle("Actual vs. Predicted Coating Properties on Unseen Test Partition (20%)", fontsize=14, y=1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Actual vs Predicted -> {save_path}")

    return fig


def plot_residuals(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Residuals (Error = Actual - Predicted) vs Predicted to diagnose
    homoscedasticity, non-linearity, and potential model bias.
    """
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(4.2 * n_targets, 4.2))

    for i, tgt in enumerate(target_names):
        ax = axes[i]
        true_vals = y_true[tgt].values
        pred_vals = y_pred[:, i]
        residuals = true_vals - pred_vals

        # Residual scatter
        ax.scatter(pred_vals, residuals, alpha=0.6, color="#2ca02c", edgecolor="black", s=40)
        ax.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.8)

        ax.set_title(f"{tgt} Residuals\nMean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f}", fontweight="bold", fontsize=11)
        ax.set_xlabel(f"Predicted {tgt}")
        ax.set_ylabel("Residual (Actual - Predicted)")

    fig.suptitle("Residual Diagnostics on Test Set (Zero Mean & Homoscedasticity Check)", fontsize=14, y=1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Residual diagnostics -> {save_path}")

    return fig


def compute_permutation_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_names: List[str],
    target_names: List[str],
    n_repeats: int = 15,
    random_state: int = 42
) -> Dict[str, pd.Series]:
    """
    Compute model-agnostic Permutation Feature Importance for each target property.
    
    Permutation importance measures the drop in model score when a feature's values
    are randomly shuffled, breaking its relationship with the target.
    """
    importance_by_target = {}

    for i, target in enumerate(target_names):
        # Wrapper to score only the i-th output target
        def target_scorer(est, X, y_col):
            preds = est.predict(X)
            if isinstance(preds, pd.DataFrame):
                preds = preds.values
            return r2_score(y_col, preds[:, i])

        res = permutation_importance(
            model,
            X_test,
            y_test[target].values,
            scoring=target_scorer,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        importance_series = pd.Series(
            res.importances_mean,
            index=feature_names
        ).sort_values(ascending=False)

        importance_by_target[target] = importance_series

    return importance_by_target


def plot_feature_importance(
    importance_by_target: Dict[str, pd.Series],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate horizontal bar charts showing relative feature importance per target property.
    """
    targets = list(importance_by_target.keys())
    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(4.5 * n_targets, 4.5))

    palette = sns.color_palette("mako", n_colors=len(next(iter(importance_by_target.values()))))

    for i, tgt in enumerate(targets):
        ax = axes[i]
        series = importance_by_target[tgt].sort_values(ascending=True)

        y_positions = np.arange(len(series))
        ax.barh(y_positions, series.values, color=palette, edgecolor="black", alpha=0.85)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f.replace("_", " ") for f in series.index], fontsize=10)
        ax.set_xlabel("Mean Decrease in R² Score")
        ax.set_title(f"Target: {tgt}", fontweight="bold", fontsize=12)

    fig.suptitle("Permutation Feature Importance Across Coating Properties", fontsize=15, y=1.04)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Feature importance -> {save_path}")

    return fig


def generate_tradeoff_candidates(
    model_bundle: Any,
    n_samples: int = 350,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate feasible random formulations strictly satisfying mass balance
    (sum = 100%) and operational bounds, and predict their coating properties.
    """
    from src.data_preprocessing import FORMULATION_INPUTS, INPUT_FEATURES
    from src.prediction import predict_properties

    rng = np.random.default_rng(random_state)

    raw_resin = rng.uniform(30.0, 60.0, size=n_samples)
    raw_pigment = rng.uniform(10.0, 38.0, size=n_samples)
    raw_solvent = rng.uniform(12.0, 28.0, size=n_samples)
    raw_additive = rng.uniform(0.5, 5.0, size=n_samples)
    raw_curing_agent = rng.uniform(1.5, 9.5, size=n_samples)

    raw_sum = raw_resin + raw_pigment + raw_solvent + raw_additive + raw_curing_agent

    resin = np.round((raw_resin / raw_sum) * 100.0, 2)
    pigment = np.round((raw_pigment / raw_sum) * 100.0, 2)
    solvent = np.round((raw_solvent / raw_sum) * 100.0, 2)
    additive = np.round((raw_additive / raw_sum) * 100.0, 2)
    curing_agent = np.round(100.0 - (resin + pigment + solvent + additive), 2)

    curing_temp = np.round(rng.uniform(60.0, 140.0, size=n_samples), 1)
    curing_time = np.round(rng.uniform(15.0, 90.0, size=n_samples), 1)

    candidates_df = pd.DataFrame({
        "Resin_Content": resin,
        "Pigment_Content": pigment,
        "Solvent_Content": solvent,
        "Additive_Content": additive,
        "Curing_Agent_Content": curing_agent,
        "Curing_Temperature": curing_temp,
        "Curing_Time": curing_time
    })

    # Predict properties for all candidates
    preds_df = predict_properties(candidates_df, model_bundle=model_bundle)

    # Combine inputs and predicted targets
    full_df = pd.concat([candidates_df, preds_df], axis=1)
    return full_df


def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = "Flexibility",
    y_col: str = "Hardness"
) -> pd.DataFrame:
    """
    Identify non-dominated Pareto optimal solutions maximizing both x_col and y_col.
    """
    points = df[[x_col, y_col]].values
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # If point j dominates point i (higher or equal in both, strictly higher in one)
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1]) and \
                   (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1]):
                    is_pareto[i] = False
                    break

    pareto_df = df[is_pareto].copy().sort_values(by=x_col).reset_index(drop=True)
    return pareto_df


def evaluate_experimental_validation(
    ai_predicted: Dict[str, float],
    experimental_measured: Dict[str, float]
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compare AI predicted properties with experimentally measured laboratory values.

    Parameters
    ----------
    ai_predicted : Dict[str, float]
        Dictionary of AI predicted properties.
    experimental_measured : Dict[str, float]
        Dictionary of bench-measured properties.

    Returns
    -------
    Tuple[pd.DataFrame, float, float]
        (comparison_df, overall_mae, overall_mape)
    """
    from src.visualization import FEATURE_UNITS

    common_props = [p for p in ai_predicted if p in experimental_measured]
    if not common_props:
        raise ValueError("No common properties found between AI predictions and experimental inputs.")

    records = []
    abs_errors = []
    pct_errors = []

    for prop in common_props:
        p_val = float(ai_predicted[prop])
        e_val = float(experimental_measured[prop])

        abs_err = abs(e_val - p_val)
        pct_err = (abs_err / e_val) * 100.0 if e_val != 0.0 else 0.0

        abs_errors.append(abs_err)
        pct_errors.append(pct_err)

        records.append({
            "Property": prop,
            "AI-Predicted": round(p_val, 2),
            "Experimental": round(e_val, 2),
            "Unit": FEATURE_UNITS.get(prop, ""),
            "Absolute Error": round(abs_err, 2),
            "Percentage Error (%)": round(pct_err, 2)
        })

    val_df = pd.DataFrame(records)
    overall_mae = float(np.mean(abs_errors))
    overall_mape = float(np.mean(pct_errors))

    return val_df, overall_mae, overall_mape

