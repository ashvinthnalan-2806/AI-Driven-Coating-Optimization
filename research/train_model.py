"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Script: train_model.py
Description: End-to-end model training, multi-algorithm benchmarking,
             evaluation, permutation feature importance, and model persistence.
=============================================================================
"""

import os
import sys
import pandas as pd

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.data_preprocessing import (
    load_dataset,
    validate_coating_data,
    preprocess_data,
    split_data,
    INPUT_FEATURES,
    TARGET_PROPERTIES
)
from src.model_training import (
    train_and_compare_models,
    save_model_bundle
)
from src.evaluation import (
    compute_property_metrics_table,
    plot_actual_vs_predicted,
    plot_residuals,
    compute_permutation_feature_importance,
    plot_feature_importance
)


def main():
    data_path = os.path.join(SCRIPT_DIR, "data", "coating_data.csv")
    models_dir = os.path.join(SCRIPT_DIR, "models")
    plots_dir = os.path.join(SCRIPT_DIR, "outputs", "plots")
    reports_dir = os.path.join(SCRIPT_DIR, "outputs", "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 75)
    print("STAGE 3: MULTI-OUTPUT MACHINE LEARNING BENCHMARKING & EVALUATION")
    print("=" * 75)

    # 1. Ingestion & Preprocessing
    print(f"\n[1/6] Ingesting and validating dataset from: {data_path}")
    raw_df = load_dataset(data_path)
    val_report = validate_coating_data(raw_df)
    if val_report["status"] == "FAIL":
        print("[ABORT] Dataset failed validation audit. Halt.")
        return

    clean_df, _ = preprocess_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

    # 2. Multi-Model Benchmarking
    print("\n[2/6] Benchmarking Candidate Algorithms...")
    best_name, best_model, comparison_df, detailed_results = train_and_compare_models(
        X_train, y_train, X_test, y_test, TARGET_PROPERTIES, random_state=42
    )

    print("\n" + "=" * 75)
    print("MODEL COMPARISON BENCHMARK (Evaluated on Unseen 20% Test Partition)")
    print("=" * 75)
    print(comparison_df.to_string(index=False))
    print("=" * 75)

    # Save comparison report
    comp_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"[Saved] Model comparison report -> {comp_path}")

    # 3. Selected Model In-Depth Evaluation
    print(f"\n[3/6] Generating Property-by-Property Metrics for Selected Model: '{best_name}'")
    y_test_pred = best_model.predict(X_test)
    metrics_table = compute_property_metrics_table(y_test, y_test_pred, TARGET_PROPERTIES)

    print("\n" + "-" * 55)
    print(f"Detailed Test Metrics for {best_name}:")
    print("-" * 55)
    print(f"{'Property':<15} {'R²':>10} {'MAE':>12} {'RMSE':>12}")
    print("-" * 55)
    for _, row in metrics_table.iterrows():
        print(f"{row['Property']:<15} {row['R²']:>10.4f} {row['MAE']:>12.3f} {row['RMSE']:>12.3f}")
    print("-" * 55)

    metrics_csv_path = os.path.join(reports_dir, "best_model_metrics.csv")
    metrics_table.to_csv(metrics_csv_path, index=False)
    print(f"[Saved] Detailed metrics -> {metrics_csv_path}")

    # 4. Evaluation Visualizations
    print("\n[4/6] Generating Parity (Actual vs Predicted) & Residual Plots...")
    parity_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    plot_actual_vs_predicted(y_test, y_test_pred, TARGET_PROPERTIES, save_path=parity_path)

    residual_path = os.path.join(plots_dir, "residual_analysis.png")
    plot_residuals(y_test, y_test_pred, TARGET_PROPERTIES, save_path=residual_path)

    # 5. Permutation Feature Importance
    print("\n[5/6] Computing Permutation Feature Importance Across Coating Properties...")
    importance_by_target = compute_permutation_feature_importance(
        best_model, X_test, y_test, INPUT_FEATURES, TARGET_PROPERTIES, n_repeats=15, random_state=42
    )

    fi_df = pd.DataFrame(importance_by_target)
    fi_csv_path = os.path.join(reports_dir, "feature_importance.csv")
    fi_df.to_csv(fi_csv_path)
    print(f"[Saved] Feature importance CSV -> {fi_csv_path}")

    fi_plot_path = os.path.join(plots_dir, "feature_importance.png")
    plot_feature_importance(importance_by_target, save_path=fi_plot_path)

    # 6. Model Persistence
    print("\n[6/6] Persisting Trained Model Bundle...")
    saved_model_path = os.path.join(models_dir, "best_coating_model.joblib")
    save_model_bundle(
        model=best_model,
        model_name=best_name,
        feature_names=INPUT_FEATURES,
        target_names=TARGET_PROPERTIES,
        metrics=detailed_results[best_name]["overall"],
        filepath=saved_model_path
    )

    print("\n" + "=" * 75)
    print("[SUCCESS] STAGE 3 COMPLETE: Model trained, evaluated, and saved.")
    print(f"Best Model File: {saved_model_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
