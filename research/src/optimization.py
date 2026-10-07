"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Module: src/optimization.py
Description: Constrained inverse design engine using Differential Evolution
             and normalized weighted objective functions.
=============================================================================
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, LinearConstraint

from src.data_preprocessing import (
    INPUT_FEATURES,
    FORMULATION_INPUTS,
    PROCESS_INPUTS,
    TARGET_PROPERTIES
)
from src.prediction import predict_properties
from src.model_training import load_model_bundle
from src.visualization import FEATURE_UNITS


# Standard engineering operational search bounds (User-editable)
DEFAULT_OPTIMIZATION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Resin_Content": (30.0, 60.0),          # wt%
    "Pigment_Content": (10.0, 40.0),         # wt%
    "Solvent_Content": (10.0, 30.0),         # wt%
    "Additive_Content": (0.5, 8.0),          # wt%
    "Curing_Agent_Content": (1.0, 10.0),     # wt%
    "Curing_Temperature": (50.0, 150.0),     # °C
    "Curing_Time": (10.0, 120.0),            # minutes
}

# Normalization characteristic scales (prevents large-scale properties from dominating)
PROPERTY_NORMALIZATION_SCALES: Dict[str, float] = {
    "Hardness": 100.0,
    "Adhesion": 100.0,
    "Flexibility": 100.0,
    "Gloss": 100.0,
    "Drying_Time": 120.0,
}


def build_inverse_objective_function(
    target_properties: Dict[str, float],
    property_weights: Dict[str, float],
    model: Any,
    feature_names: List[str],
    target_names: List[str],
    penalty_weight: float = 1000.0
):
    """
    Construct a scale-normalized, constrained multi-objective loss function.

    Objective = sum_i( w_i * ((y_pred_i - y_target_i) / Scale_i)^2 )
                + penalty_weight * (sum(Components) - 100.0)^2
    """
    active_targets = list(target_properties.keys())
    target_indices = [target_names.index(t) for t in active_targets]
    target_vals = np.array([target_properties[t] for t in active_targets])
    weights = np.array([property_weights.get(t, 1.0) for t in active_targets])
    scales = np.array([PROPERTY_NORMALIZATION_SCALES.get(t, 100.0) for t in active_targets])

    # Index of formulation components in input feature vector
    formulation_indices = [feature_names.index(col) for col in FORMULATION_INPUTS]

    def objective_fn(x: np.ndarray) -> float:
        # 1. Mass balance penalty (Resin + Pigment + Solvent + Additive + CuringAgent == 100.0)
        comp_sum = np.sum(x[formulation_indices])
        mass_penalty = penalty_weight * ((comp_sum - 100.0) ** 2)

        # 2. Forward model prediction (fast numpy prediction with warning suppressed)
        x_2d = x.reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            preds = model.predict(x_2d)[0]

        # 3. Normalized weighted squared error
        pred_active = preds[target_indices]
        normalized_errors = (pred_active - target_vals) / scales
        weighted_loss = np.sum(weights * (normalized_errors ** 2))

        total_loss = float(weighted_loss + mass_penalty)
        return total_loss

    return objective_fn


def inverse_design(
    target_properties: Dict[str, float],
    property_weights: Optional[Dict[str, float]] = None,
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    model_bundle: Optional[Union[str, Dict[str, Any]]] = None,
    popsize: int = 25,
    maxiter: int = 250,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform inverse formulation optimization to match user-defined target properties.

    Parameters
    ----------
    target_properties : Dict[str, float]
        Dictionary of desired coating properties, e.g.:
        {"Hardness": 85.0, "Flexibility": 80.0, "Gloss": 85.0}
    property_weights : Optional[Dict[str, float]]
        Relative importance weights for each target property (default: 1.0 each).
    custom_bounds : Optional[Dict[str, Tuple[float, float]]]
        User-specified operational search bounds. If None, uses DEFAULT_OPTIMIZATION_BOUNDS.
    model_bundle : Optional[Union[str, Dict[str, Any]]]
        Loaded model bundle or path to joblib file.
    popsize : int
        Population size factor for Differential Evolution (default: 25).
    maxiter : int
        Maximum number of evolutionary generations (default: 250).
    random_state : int
        Random seed for reproducible stochastic global search.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'optimized_formulation': Dict of optimal formulation and process inputs
        - 'predicted_properties': Dict of AI-predicted properties for that recipe
        - 'comparison_table': DataFrame comparing Target vs. Predicted with errors
        - 'optimization_metadata': Iteration count, convergence status, message.
    """
    # 1. Load model bundle
    if model_bundle is None:
        model_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "models", "best_coating_model.joblib")
        )
        model_bundle = load_model_bundle(model_path)
    elif isinstance(model_bundle, str):
        model_bundle = load_model_bundle(model_bundle)

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    target_names = model_bundle["target_names"]

    # 2. Validate requested targets
    invalid_targets = [t for t in target_properties if t not in target_names]
    if invalid_targets:
        raise ValueError(f"Unknown target property requested: {invalid_targets}. Available: {target_names}")

    if not target_properties:
        raise ValueError("Must provide at least one target property for inverse optimization.")

    # 3. Setup weights
    if property_weights is None:
        property_weights = {t: 1.0 for t in target_properties}
    else:
        for t in target_properties:
            if t not in property_weights:
                property_weights[t] = 1.0

    # 4. Resolve search bounds
    active_bounds = DEFAULT_OPTIMIZATION_BOUNDS.copy()
    if custom_bounds:
        active_bounds.update(custom_bounds)

    bounds_list = [active_bounds[feat] for feat in feature_names]

    # 5. Formulate strict linear mass constraint: sum(Formulation_Inputs) = 100.0 wt%
    # Matrix A has 1 for formulation components and 0 for temperature and time
    a_row = [1.0 if feat in FORMULATION_INPUTS else 0.0 for feat in feature_names]
    mass_constraint = LinearConstraint(A=[a_row], lb=[99.5], ub=[100.5])

    # 6. Build scale-normalized objective function
    objective_fn = build_inverse_objective_function(
        target_properties=target_properties,
        property_weights=property_weights,
        model=model,
        feature_names=feature_names,
        target_names=target_names
    )

    # 7. Execute Global Differential Evolution Optimization
    opt_result = differential_evolution(
        func=objective_fn,
        bounds=bounds_list,
        constraints=(mass_constraint,),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        mutation=(0.5, 1.2),
        recombination=0.75,
        seed=random_state,
        polish=False
    )

    # 8. Post-process raw solution and enforce strict 100.00% mass balance
    raw_solution = opt_result.x
    sol_dict = {feat: float(raw_solution[i]) for i, feat in enumerate(feature_names)}

    # Normalize formulation components strictly so sum == 100.00 wt%
    comp_sum = sum(sol_dict[c] for c in FORMULATION_INPUTS)
    for c in FORMULATION_INPUTS:
        sol_dict[c] = round((sol_dict[c] / comp_sum) * 100.0, 2)
    # Fix rounding residue on Curing_Agent
    other_sum = sum(sol_dict[c] for c in FORMULATION_INPUTS if c != "Curing_Agent_Content")
    sol_dict["Curing_Agent_Content"] = round(100.0 - other_sum, 2)

    sol_dict["Curing_Temperature"] = round(sol_dict["Curing_Temperature"], 1)
    sol_dict["Curing_Time"] = round(sol_dict["Curing_Time"], 1)

    # 9. Forward predict properties for the optimized recipe
    pred_df = predict_properties(sol_dict, model_bundle=model_bundle)
    pred_dict = dict(zip(pred_df["Property"], pred_df["Predicted Value"]))

    # 10. Build Target vs Predicted Comparison Table
    records = []
    for tgt in target_names:
        pred_val = pred_dict[tgt]
        if tgt in target_properties:
            tgt_val = target_properties[tgt]
            abs_err = abs(pred_val - tgt_val)
            pct_err = (abs_err / tgt_val) * 100.0 if tgt_val != 0 else 0.0
            status = "Targeted"
        else:
            tgt_val = np.nan
            abs_err = np.nan
            pct_err = np.nan
            status = "Co-Predicted"

        records.append({
            "Property": tgt,
            "Target Value": round(tgt_val, 2) if not np.isnan(tgt_val) else "---",
            "AI-Predicted": round(pred_val, 2),
            "Absolute Error": round(abs_err, 2) if not np.isnan(abs_err) else "---",
            "Percentage Error (%)": round(pct_err, 2) if not np.isnan(pct_err) else "---",
            "Status": status
        })

    comparison_table = pd.DataFrame(records)

    is_converged = bool(opt_result.success or (opt_result.fun < 1e-3))
    status_label = "CONVERGED (Global Minimum Found)" if is_converged else "TERMINATED (Best Feasible Found)"

    results = {
        "optimized_formulation": sol_dict,
        "predicted_properties": pred_dict,
        "comparison_table": comparison_table,
        "optimization_metadata": {
            "success": is_converged,
            "status_label": status_label,
            "message": opt_result.message,
            "nit": opt_result.nit,
            "nfev": opt_result.nfev,
            "fun": float(opt_result.fun)
        }
    }

    return results


def print_inverse_design_results(results: Dict[str, Any]) -> None:
    """Pretty-print inverse formulation optimization results to console."""
    opt_recipe = results["optimized_formulation"]
    comp_table = results["comparison_table"]
    meta = results["optimization_metadata"]

    print("=" * 75)
    print("AI INVERSE DESIGN: RECOMMENDED OPTIMIZED COATING RECIPE")
    print("=" * 75)
    print(f"Convergence Status : {meta.get('status_label', 'SUCCESS')}")
    print(f"Evolution Iterations: {meta['nit']} generations (Function evaluations: {meta['nfev']})")
    print(f"Final Objective Loss: {meta['fun']:.6f}")
    print("-" * 75)
    print("RECOMMENDED FORMULATION & BAKING PROCESS CONDITIONS:")
    print("-" * 75)
    for c in FORMULATION_INPUTS:
        print(f"  {c.replace('_', ' '):<28}: {opt_recipe[c]:>6.2f} wt%")
    f_sum = sum(opt_recipe[c] for c in FORMULATION_INPUTS)
    print(f"  {'Formulation Mass Conservation':<28}: {f_sum:>6.2f} wt%  [Strict 100% Satisfied]")
    print("-" * 75)
    for p in PROCESS_INPUTS:
        unit = FEATURE_UNITS.get(p, "")
        print(f"  {p.replace('_', ' '):<28}: {opt_recipe[p]:>6.2f} {unit}")

    print("\n" + "=" * 75)
    print("TARGET VS. AI-PREDICTED PROPERTY COMPARISON:")
    print("=" * 75)
    print(comp_table.to_string(index=False))
    print("=" * 75)
