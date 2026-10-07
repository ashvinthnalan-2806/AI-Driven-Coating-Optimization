"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Module: src/data_preprocessing.py
Description: Rigorous data validation, preprocessing, and leak-free splitting
             for coating formulation and process parameters.
=============================================================================
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------------------------------------
# Global Schema & Operational Boundaries
# ---------------------------------------------------------------------------
FORMULATION_INPUTS: List[str] = [
    "Resin_Content",
    "Pigment_Content",
    "Solvent_Content",
    "Additive_Content",
    "Curing_Agent_Content",
]

PROCESS_INPUTS: List[str] = [
    "Curing_Temperature",
    "Curing_Time",
]

INPUT_FEATURES: List[str] = FORMULATION_INPUTS + PROCESS_INPUTS

TARGET_PROPERTIES: List[str] = [
    "Hardness",
    "Adhesion",
    "Flexibility",
    "Gloss",
    "Drying_Time",
]

# Physical and Chemical Engineering Boundaries (Min, Max)
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Resin_Content": (20.0, 70.0),          # wt%
    "Pigment_Content": (5.0, 50.0),         # wt%
    "Solvent_Content": (5.0, 45.0),         # wt%
    "Additive_Content": (0.0, 15.0),        # wt%
    "Curing_Agent_Content": (0.5, 15.0),    # wt%
    "Curing_Temperature": (40.0, 180.0),    # °C
    "Curing_Time": (5.0, 180.0),            # minutes
}

TARGET_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Hardness": (0.0, 100.0),               # Koenig / Shore index
    "Adhesion": (0.0, 100.0),               # Cross-hatch rating index
    "Flexibility": (0.0, 100.0),            # Mandrel bend / elongation index
    "Gloss": (0.0, 100.0),                  # GU @ 60°
    "Drying_Time": (1.0, 240.0),            # minutes
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Safely load coating formulation dataset from CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def validate_coating_data(
    df: pd.DataFrame,
    strict_mass_balance: bool = True,
    mass_tolerance: float = 0.5
) -> Dict[str, Any]:
    """
    Rigorously validate schema, datatypes, boundaries, and mass conservation.
    Does NOT silently alter or delete data. Returns a comprehensive audit report.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to audit.
    strict_mass_balance : bool
        Whether to enforce sum(formulation_inputs) == 100%.
    mass_tolerance : float
        Allowed deviation from 100.0 wt% (default: ±0.5%).

    Returns
    -------
    Dict[str, Any]
        Validation report dictionary with status, errors, and warnings.
    """
    report: Dict[str, Any] = {
        "status": "PASS",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_columns": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "dtype_issues": {},
        "out_of_bounds_features": {},
        "out_of_bounds_targets": {},
        "mass_balance_violations": 0,
        "messages": []
    }

    # 1. Column existence verification
    all_required_cols = INPUT_FEATURES + TARGET_PROPERTIES
    missing_cols = [c for c in all_required_cols if c not in df.columns]
    if missing_cols:
        report["missing_columns"] = missing_cols
        report["status"] = "FAIL"
        report["messages"].append(f"CRITICAL: Missing required columns: {missing_cols}")
        return report  # Halt further checks if key columns are missing

    # 2. Data types and non-numeric validation
    for col in all_required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            report["dtype_issues"][col] = str(df[col].dtype)
            report["status"] = "FAIL"
            report["messages"].append(f"ERROR: Column '{col}' is not numeric ({df[col].dtype}).")

    # 3. Missing values check
    null_counts = df[all_required_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        report["missing_values"] = cols_with_nulls
        report["messages"].append(f"WARNING: Found missing values in columns: {cols_with_nulls}")

    # 4. Duplicate rows check
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        report["duplicate_rows"] = int(duplicates)
        report["messages"].append(f"WARNING: Found {duplicates} duplicate row(s).")

    # 5. Feature range boundaries check
    for feat, (b_min, b_max) in FEATURE_BOUNDS.items():
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            invalid_mask = (df[feat] < b_min) | (df[feat] > b_max)
            count = int(invalid_mask.sum())
            if count > 0:
                report["out_of_bounds_features"][feat] = {
                    "count": count,
                    "expected_range": (b_min, b_max),
                    "min_observed": float(df[feat].min()),
                    "max_observed": float(df[feat].max())
                }
                report["messages"].append(
                    f"WARNING: '{feat}' has {count} sample(s) outside expected range [{b_min}, {b_max}]."
                )

    # 6. Target range boundaries check
    for tgt, (t_min, t_max) in TARGET_BOUNDS.items():
        if tgt in df.columns and pd.api.types.is_numeric_dtype(df[tgt]):
            invalid_mask = (df[tgt] < t_min) | (df[tgt] > t_max)
            count = int(invalid_mask.sum())
            if count > 0:
                report["out_of_bounds_targets"][tgt] = {
                    "count": count,
                    "expected_range": (t_min, t_max),
                    "min_observed": float(df[tgt].min()),
                    "max_observed": float(df[tgt].max())
                }
                report["messages"].append(
                    f"WARNING: Target '{tgt}' has {count} sample(s) outside expected range [{t_min}, {t_max}]."
                )

    # 7. Physical Mass Balance Constraint (Sum of formulation wt% = 100.0%)
    formulation_sums = df[FORMULATION_INPUTS].sum(axis=1)
    violations = np.abs(formulation_sums - 100.0) > mass_tolerance
    violation_count = int(violations.sum())
    report["mass_balance_violations"] = violation_count

    if violation_count > 0:
        max_diff = float(np.abs(formulation_sums - 100.0).max())
        msg = (
            f"MASS BALANCE WARNING: {violation_count} formulation(s) deviate from 100.0 wt% "
            f"by more than {mass_tolerance}% (Max deviation = {max_diff:.3f}%)."
        )
        report["messages"].append(msg)
        if strict_mass_balance and violation_count > len(df) * 0.1:
            report["status"] = "FAIL"

    if report["status"] == "PASS" and not report["messages"]:
        report["messages"].append("SUCCESS: All schema, datatype, range, and mass balance audits passed.")

    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """Pretty-print the data validation report to terminal."""
    print("=" * 70)
    print("DATA INTEGRITY & VALIDATION REPORT")
    print("=" * 70)
    print(f"Overall Audit Status: [{report['status']}]")
    print(f"Total Rows Analyzed : {report['total_rows']}")
    print(f"Total Columns       : {report['total_columns']}")
    print(f"Duplicate Rows      : {report['duplicate_rows']}")
    print(f"Mass Balance Errors : {report['mass_balance_violations']}")

    if report["missing_values"]:
        print("\nMissing Values Detected:")
        for col, count in report["missing_values"].items():
            pct = (count / report["total_rows"]) * 100
            print(f"  - {col:25s}: {count} rows ({pct:.2f}%)")

    if report["out_of_bounds_features"]:
        print("\nFeature Range Warnings:")
        for feat, data in report["out_of_bounds_features"].items():
            print(f"  - {feat:22s}: {data['count']} rows out of {data['expected_range']} (Observed: [{data['min_observed']:.2f}, {data['max_observed']:.2f}])")

    if report["out_of_bounds_targets"]:
        print("\nTarget Range Warnings:")
        for tgt, data in report["out_of_bounds_targets"].items():
            print(f"  - {tgt:22s}: {data['count']} rows out of {data['expected_range']} (Observed: [{data['min_observed']:.2f}, {data['max_observed']:.2f}])")

    print("\nDetailed Diagnostic Messages:")
    for msg in report["messages"]:
        print(f"  * {msg}")
    print("=" * 70)


def preprocess_data(
    df: pd.DataFrame,
    handle_missing: str = "median",
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and preprocess dataset while preserving data provenance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    handle_missing : str
        Strategy for missing values ('median', 'mean', or 'drop').
    drop_duplicates : bool
        Whether to drop exact duplicate observations.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and preprocessing log summary.
    """
    clean_df = df.copy()
    log: Dict[str, Any] = {
        "initial_rows": len(df),
        "dropped_duplicates": 0,
        "imputed_cells": 0,
        "final_rows": 0,
    }

    # 1. Handle Duplicates
    if drop_duplicates:
        dups = clean_df.duplicated().sum()
        if dups > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            log["dropped_duplicates"] = int(dups)

    # 2. Handle Missing Values
    null_cells = clean_df[INPUT_FEATURES + TARGET_PROPERTIES].isnull().sum().sum()
    if null_cells > 0:
        log["imputed_cells"] = int(null_cells)
        if handle_missing == "drop":
            clean_df = clean_df.dropna(subset=INPUT_FEATURES + TARGET_PROPERTIES).reset_index(drop=True)
        else:
            # Impute numerical columns with column median to prevent outlier distortion
            for col in INPUT_FEATURES + TARGET_PROPERTIES:
                if clean_df[col].isnull().sum() > 0:
                    fill_val = clean_df[col].median() if handle_missing == "median" else clean_df[col].mean()
                    clean_df[col] = clean_df[col].fillna(fill_val)

    log["final_rows"] = len(clean_df)
    return clean_df, log


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leak-free feature/target split and train/test partition.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    test_size : float
        Fraction of samples for testing (default: 0.20 = 20%).
    random_state : int
        Seed for deterministic splitting.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        X_train, X_test, y_train, y_test
    """
    X = df[INPUT_FEATURES].copy()
    y = df[TARGET_PROPERTIES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor_pipeline(scaling: str = "none") -> ColumnTransformer:
    """
    Create an scikit-learn preprocessing ColumnTransformer.
    
    Tree ensembles (Random Forest, Extra Trees, Gradient Boosting) are invariant
    to monotonic feature scaling. However, scaling ('standard' or 'minmax') can be
    enabled for linear models or distance-based baselines.

    Parameters
    ----------
    scaling : str
        'none', 'standard', or 'minmax'.

    Returns
    -------
    ColumnTransformer
        Sklearn preprocessor object ready for inclusion in a Pipeline.
    """
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = "passthrough"

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, INPUT_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor
