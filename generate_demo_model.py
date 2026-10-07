"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Script: generate_demo_model.py
Description: Generates a demonstration multi-output regression model and
             accompanying metadata for testing the Flask web application.
=============================================================================

IMPORTANT SCIENTIFIC NOTICE:
This model is trained on a clearly labelled DEMONSTRATION / SYNTHETIC dataset
derived from established polymer science principles. It is intended for testing
the web interface and software pipeline. Replace with an experimentally trained
model before industrial application.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

INPUT_FEATURES = [
    "Resin_Content",
    "Pigment_Content",
    "Solvent_Content",
    "Additive_Content",
    "Curing_Agent_Content",
    "Curing_Temperature",
    "Curing_Time"
]

TARGET_PROPERTIES = [
    "Hardness",
    "Adhesion",
    "Flexibility",
    "Gloss",
    "Drying_Time"
]

PROPERTY_UNITS = {
    "Hardness": "Index (0-100)",
    "Adhesion": "Index (0-100)",
    "Flexibility": "Index (0-100)",
    "Gloss": "GU @ 60°",
    "Drying_Time": "minutes"
}


def generate_synthetic_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic demonstration coating dataset."""
    rng = np.random.default_rng(random_state)

    raw_resin = rng.uniform(34.0, 56.0, size=n_samples)
    raw_pigment = rng.uniform(12.0, 32.0, size=n_samples)
    raw_solvent = rng.uniform(14.0, 26.0, size=n_samples)
    raw_additive = rng.uniform(0.8, 4.5, size=n_samples)
    raw_curing_agent = rng.uniform(2.0, 8.5, size=n_samples)

    raw_sum = raw_resin + raw_pigment + raw_solvent + raw_additive + raw_curing_agent

    resin = np.round((raw_resin / raw_sum) * 100.0, 2)
    pigment = np.round((raw_pigment / raw_sum) * 100.0, 2)
    solvent = np.round((raw_solvent / raw_sum) * 100.0, 2)
    additive = np.round((raw_additive / raw_sum) * 100.0, 2)
    curing_agent = np.round(100.0 - (resin + pigment + solvent + additive), 2)

    curing_temp = np.round(rng.uniform(60.0, 140.0, size=n_samples), 1)
    curing_time = np.round(rng.uniform(15.0, 90.0, size=n_samples), 1)

    # Polymer science response modeling
    hardness = 15.0 + 2.8 * curing_agent + 0.75 * pigment + 0.22 * curing_temp + 0.12 * curing_time - 0.35 * solvent + 0.015 * (curing_agent * curing_temp) + rng.normal(0, 1.2, n_samples)
    adhesion = 48.0 + 0.55 * resin + 2.2 * additive + 0.8 * curing_agent - 0.3 * solvent - 0.003 * ((curing_temp - 100.0) ** 2) + rng.normal(0, 1.0, n_samples)
    flexibility = 38.0 + 1.15 * resin - 2.9 * curing_agent - 0.85 * pigment - 0.12 * curing_time - 0.08 * curing_temp + rng.normal(0, 1.2, n_samples)
    gloss = 92.0 - 1.45 * pigment + 0.35 * resin + 1.8 * additive - 0.15 * np.abs(solvent - 20.0) + rng.normal(0, 1.2, n_samples)
    drying_time = 115.0 - 0.48 * curing_temp - 0.22 * curing_time - 3.8 * curing_agent + 1.1 * solvent + rng.normal(0, 1.0, n_samples)

    df = pd.DataFrame({
        "Resin_Content": resin,
        "Pigment_Content": pigment,
        "Solvent_Content": solvent,
        "Additive_Content": additive,
        "Curing_Agent_Content": curing_agent,
        "Curing_Temperature": curing_temp,
        "Curing_Time": curing_time,
        "Hardness": np.clip(np.round(hardness, 2), 25.0, 98.0),
        "Adhesion": np.clip(np.round(adhesion, 2), 30.0, 99.0),
        "Flexibility": np.clip(np.round(flexibility, 2), 15.0, 95.0),
        "Gloss": np.clip(np.round(gloss, 2), 10.0, 96.0),
        "Drying_Time": np.clip(np.round(drying_time, 1), 10.0, 120.0)
    })
    return df


def main():
    print("=" * 70)
    print("GENERATING DEMONSTRATION ML MODEL FOR WEB APPLICATION")
    print("=" * 70)

    # 1. Dataset Generation
    df = generate_synthetic_data(n_samples=500, random_state=42)
    data_csv_path = os.path.join(DATA_DIR, "coating_data.csv")
    df.to_csv(data_csv_path, index=False)
    print(f"[OK] Saved demonstration dataset -> {data_csv_path}")

    # 2. Partition Data
    X = df[INPUT_FEATURES]
    y = df[TARGET_PROPERTIES]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Train Preprocessing Pipeline & Model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Preprocessing Pipeline
    prep_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    joblib.dump(scaler, prep_path)
    print(f"[OK] Saved preprocessing pipeline -> {prep_path}")

    # Multi-output Gradient Boosting Regressor
    base_est = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )
    model = MultiOutputRegressor(base_est)
    print("Fitting Multi-Output Gradient Boosting Regressor...")
    model.fit(X_train_scaled, y_train)

    # Also build a unified combined pipeline
    unified_pipeline = Pipeline([
        ("scaler", scaler),
        ("regressor", model)
    ])

    model_path = os.path.join(MODELS_DIR, "coating_model.joblib")
    joblib.dump(unified_pipeline, model_path)
    print(f"[OK] Saved trained model pipeline -> {model_path}")

    # 4. Compute Test Metrics
    y_pred = unified_pipeline.predict(X_test)
    metrics_per_target = {}
    r2_list, mae_list, rmse_list = [], [], []

    for i, tgt in enumerate(TARGET_PROPERTIES):
        y_true_col = y_test[tgt].values
        y_pred_col = y_pred[:, i]
        r2 = float(r2_score(y_true_col, y_pred_col))
        mae = float(mean_absolute_error(y_true_col, y_pred_col))
        rmse = float(root_mean_squared_error(y_true_col, y_pred_col))

        metrics_per_target[tgt] = {
            "R2": round(r2, 4),
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "Unit": PROPERTY_UNITS[tgt]
        }
        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)

    overall_metrics = {
        "Mean_R2": round(float(np.mean(r2_list)), 4),
        "Mean_MAE": round(float(np.mean(mae_list)), 3),
        "Mean_RMSE": round(float(np.mean(rmse_list)), 3)
    }

    # 5. Save Metadata
    metadata = {
        "model_name": "Gradient Boosting Regressor (Multi-Output)",
        "task": "Multi-output Regression",
        "is_demo": True,
        "demo_disclaimer": "DEMO MODEL — Replace with experimentally trained model.",
        "input_features": INPUT_FEATURES,
        "target_properties": TARGET_PROPERTIES,
        "target_units": PROPERTY_UNITS,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "overall_metrics": overall_metrics,
        "per_target_metrics": metrics_per_target,
        "created_at": pd.Timestamp.now().isoformat()
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved model metadata -> {metadata_path}")

    print("\nModel Evaluation Summary on Unseen Test Partition:")
    print(f"  Mean R²: {overall_metrics['Mean_R2']:.4f} | Mean MAE: {overall_metrics['Mean_MAE']:.3f} | Mean RMSE: {overall_metrics['Mean_RMSE']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
