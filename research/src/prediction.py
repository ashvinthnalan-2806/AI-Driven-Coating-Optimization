"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Module: src/prediction.py
Description: Forward prediction module mapping coating formulation and baking
             process parameters to predicted performance properties.
=============================================================================
"""

import os
from typing import Dict, Union, Any, Optional
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    INPUT_FEATURES,
    FORMULATION_INPUTS,
    PROCESS_INPUTS,
    TARGET_PROPERTIES,
    FEATURE_BOUNDS,
    TARGET_BOUNDS
)
from src.model_training import load_model_bundle
from src.visualization import FEATURE_UNITS


DEFAULT_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "best_coating_model.joblib")
)


def validate_input_formulation(
    formulation: Dict[str, float],
    strict_sum: bool = False,
    tolerance: float = 1.0
) -> Dict[str, Any]:
    """
    Validate that an input formulation dictionary contains all required
    variables, checks mass conservation sum, and checks boundary limits.

    Parameters
    ----------
    formulation : Dict[str, float]
        Input dictionary of parameter values.
    strict_sum : bool
        If True, raises error if formulation components don't sum to 100%.
    tolerance : float
        Allowed deviation from 100.0 wt%.

    Returns
    -------
    Dict[str, Any]
        Validation status report with warnings and computed sum.
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "component_sum": 0.0
    }

    # 1. Missing keys check
    missing = [feat for feat in INPUT_FEATURES if feat not in formulation]
    if missing:
        report["valid"] = False
        report["errors"].append(f"Missing required parameter(s): {missing}")
        return report

    # 2. Check formulation mass sum
    comp_sum = sum(formulation[c] for c in FORMULATION_INPUTS)
    report["component_sum"] = round(comp_sum, 2)
    if abs(comp_sum - 100.0) > tolerance:
        msg = f"Formulation components sum to {comp_sum:.2f} wt% (Expected ~100.00 wt%)."
        if strict_sum:
            report["valid"] = False
            report["errors"].append(msg)
        else:
            report["warnings"].append(msg)

    # 3. Check operational range bounds
    for feat in INPUT_FEATURES:
        val = formulation[feat]
        b_min, b_max = FEATURE_BOUNDS[feat]
        if val < b_min or val > b_max:
            report["warnings"].append(
                f"Parameter '{feat}' value {val} is outside typical range [{b_min}, {b_max}]."
            )

    return report


def predict_properties(
    formulation: Union[Dict[str, float], pd.DataFrame],
    model_bundle: Optional[Union[str, Dict[str, Any]]] = None,
    enforce_physical_limits: bool = True
) -> pd.DataFrame:
    """
    Forward inference engine: Predict coating properties for a given formulation recipe.

    Parameters
    ----------
    formulation : Union[Dict[str, float], pd.DataFrame]
        Dictionary or DataFrame containing the 7 input features:
        - Resin_Content (wt%)
        - Pigment_Content (wt%)
        - Solvent_Content (wt%)
        - Additive_Content (wt%)
        - Curing_Agent_Content (wt%)
        - Curing_Temperature (°C)
        - Curing_Time (minutes)
    model_bundle : Optional[Union[str, Dict[str, Any]]]
        Loaded bundle dictionary or filepath to .joblib model bundle.
        If None, loads the default saved model at 'models/best_coating_model.joblib'.
    enforce_physical_limits : bool
        Whether to clip predicted properties within physically realistic boundaries.

    Returns
    -------
    pd.DataFrame
        Formatted results table containing Target Property, Predicted Value,
        Physical Unit, and Engineering Interpretation.
    """
    # 1. Resolve model bundle
    if model_bundle is None:
        model_bundle = load_model_bundle(DEFAULT_MODEL_PATH)
    elif isinstance(model_bundle, str):
        model_bundle = load_model_bundle(model_bundle)

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    target_names = model_bundle["target_names"]

    # 2. Format input into 2D DataFrame
    if isinstance(formulation, dict):
        # Validate formulation
        val_res = validate_input_formulation(formulation)
        if not val_res["valid"]:
            raise ValueError(f"Input formulation validation failed: {val_res['errors']}")
        df_input = pd.DataFrame([formulation])[feature_names]
    elif isinstance(formulation, pd.DataFrame):
        df_input = formulation[feature_names].copy()
    else:
        raise TypeError("Formulation must be a dictionary or pandas DataFrame.")

    # 3. Model Inference
    raw_preds = model.predict(df_input)
    if isinstance(raw_preds, pd.DataFrame):
        raw_preds = raw_preds.values

    # Handle single or multiple rows
    is_single_recipe = (len(df_input) == 1)

    if is_single_recipe:
        preds = raw_preds[0]
        results = []

        for i, tgt in enumerate(target_names):
            val = float(preds[i])
            if enforce_physical_limits:
                t_min, t_max = TARGET_BOUNDS[tgt]
                val = float(np.clip(val, t_min, t_max))

            results.append({
                "Property": tgt,
                "Predicted Value": round(val, 2),
                "Unit": FEATURE_UNITS.get(tgt, ""),
                "Scale Range": f"[{TARGET_BOUNDS[tgt][0]:.0f} - {TARGET_BOUNDS[tgt][1]:.0f}]"
            })

        return pd.DataFrame(results)

    else:
        # Multiple formulations batch inference
        batch_df = pd.DataFrame(raw_preds, columns=target_names)
        if enforce_physical_limits:
            for tgt in target_names:
                t_min, t_max = TARGET_BOUNDS[tgt]
                batch_df[tgt] = np.clip(batch_df[tgt], t_min, t_max)
        return batch_df.round(2)


def print_prediction_table(
    formulation: Dict[str, float],
    predictions_df: pd.DataFrame
) -> None:
    """
    Format and print a clean engineering summary table of input parameters
    and predicted coating properties.
    """
    print("\n" + "=" * 70)
    print("COATING FORWARD PROPERTY PREDICTION REPORT")
    print("=" * 70)
    print("INPUT RECIPE & PROCESS PARAMETERS:")
    print("-" * 70)
    for feat in FORMULATION_INPUTS:
        print(f"  {feat.replace('_', ' '):<28}: {formulation[feat]:>6.2f} wt%")
    comp_sum = sum(formulation[c] for c in FORMULATION_INPUTS)
    print(f"  {'Total Formulation Sum':<28}: {comp_sum:>6.2f} wt%")
    print("-" * 70)
    for proc in PROCESS_INPUTS:
        unit = FEATURE_UNITS.get(proc, "")
        print(f"  {proc.replace('_', ' '):<28}: {formulation[proc]:>6.2f} {unit}")

    print("\n" + "=" * 70)
    print("AI-PREDICTED COATING PROPERTIES:")
    print("=" * 70)
    print(f"{'Property':<18} {'Predicted Value':>18} {'Unit':<15} {'Operational Range':<15}")
    print("-" * 70)
    for _, row in predictions_df.iterrows():
        print(f"{row['Property']:<18} {row['Predicted Value']:>18.2f} {row['Unit']:<15} {row['Scale Range']:<15}")
    print("=" * 70)
