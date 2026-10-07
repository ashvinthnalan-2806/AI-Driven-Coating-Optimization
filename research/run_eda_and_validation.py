"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Script: run_eda_and_validation.py
Description: Executes Stage 2: Data validation, preprocessing, leak-free split,
             and publication-quality Exploratory Data Analysis (EDA).
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
    print_validation_report,
    preprocess_data,
    split_data,
    INPUT_FEATURES,
    TARGET_PROPERTIES
)
from src.visualization import (
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_coating_domain_relationships,
    export_eda_summary_statistics
)


def main():
    data_path = os.path.join(SCRIPT_DIR, "data", "coating_data.csv")
    plots_dir = os.path.join(SCRIPT_DIR, "outputs", "plots")
    reports_dir = os.path.join(SCRIPT_DIR, "outputs", "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 75)
    print("STAGE 2: DATA VALIDATION, PREPROCESSING & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 75)

    # 1. Load Dataset
    print(f"\n[Step 1] Ingesting dataset from: {data_path}")
    raw_df = load_dataset(data_path)
    print(f"Dataset Loaded Successfully: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print("\nDataset Preview (First 5 Rows):")
    print(raw_df.head())

    # 2. Data Validation
    print("\n[Step 2] Performing Comprehensive Schema & Domain Boundary Audit...")
    validation_report = validate_coating_data(raw_df, strict_mass_balance=True)
    print_validation_report(validation_report)

    if validation_report["status"] == "FAIL":
        print("\n[ABORT] Critical validation errors detected. Resolve schema issues before proceeding.")
        return

    # 3. Data Preprocessing & Provable Provenance
    print("\n[Step 3] Running Data Cleaning & Preprocessing Pipeline...")
    clean_df, prep_log = preprocess_data(raw_df, handle_missing="median", drop_duplicates=True)
    print(f"  - Initial Rows: {prep_log['initial_rows']}")
    print(f"  - Dropped Duplicates: {prep_log['dropped_duplicates']}")
    print(f"  - Imputed Values: {prep_log['imputed_cells']}")
    print(f"  - Cleaned Dataset Shape: {clean_df.shape}")

    # 4. Leakage-Free Train/Test Split
    print("\n[Step 4] Performing Leakage-Free Train/Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)
    print(f"  - Training Features (X_train) : {X_train.shape}")
    print(f"  - Testing Features  (X_test)  : {X_test.shape}")
    print(f"  - Training Targets  (y_train) : {y_train.shape}")
    print(f"  - Testing Targets   (y_test)  : {y_test.shape}")
    print(f"  - Feature Columns: {list(X_train.columns)}")
    print(f"  - Target Columns : {list(y_train.columns)}")

    # 5. Exploratory Data Analysis (EDA) Plots & Summary
    print("\n[Step 5] Generating Publication-Quality EDA Visualizations...")

    dist_plot_path = os.path.join(plots_dir, "eda_distributions.png")
    plot_feature_distributions(clean_df, save_path=dist_plot_path)

    corr_plot_path = os.path.join(plots_dir, "eda_correlation_matrix.png")
    plot_correlation_heatmap(clean_df, save_path=corr_plot_path)

    domain_plot_path = os.path.join(plots_dir, "eda_input_vs_properties.png")
    plot_coating_domain_relationships(clean_df, save_path=domain_plot_path)

    summary_csv_path = os.path.join(reports_dir, "eda_summary.csv")
    summary_df = export_eda_summary_statistics(clean_df, save_path=summary_csv_path)

    print("\nSummary Statistics Table Preview:")
    print(summary_df)

    print("\n" + "=" * 75)
    print("[SUCCESS] Stage 2 Pipeline Completed Successfully!")
    print(f"Plots saved to: {plots_dir}")
    print(f"Reports saved to: {reports_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
