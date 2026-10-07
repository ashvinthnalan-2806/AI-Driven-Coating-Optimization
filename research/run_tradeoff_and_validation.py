"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Script: run_tradeoff_and_validation.py
Description: Executes Stage 6: Hardness vs. Flexibility Trade-off analysis
             (Pareto Frontier) and Experimental Bench Validation comparison.
=============================================================================
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.model_training import load_model_bundle
from src.prediction import predict_properties
from src.evaluation import (
    generate_tradeoff_candidates,
    compute_pareto_frontier,
    evaluate_experimental_validation
)
from src.visualization import (
    plot_hardness_flexibility_tradeoff,
    plot_ai_vs_experimental_comparison
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Hardness-Flexibility trade-off and experimental validation."
    )
    # Optional lab experimental inputs
    parser.add_argument("--exp_hardness", type=float, default=83.5, help="Lab measured Hardness")
    parser.add_argument("--exp_adhesion", type=float, default=76.8, help="Lab measured Adhesion")
    parser.add_argument("--exp_flexibility", type=float, default=43.2, help="Lab measured Flexibility")
    parser.add_argument("--exp_gloss", type=float, default=81.4, help="Lab measured Gloss")
    parser.add_argument("--exp_drying_time", type=float, default=52.0, help="Lab measured Drying Time")

    args = parser.parse_args()

    plots_dir = os.path.join(SCRIPT_DIR, "outputs", "plots")
    reports_dir = os.path.join(SCRIPT_DIR, "outputs", "reports")
    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 75)
    print("STAGE 6: HARDNESS-FLEXIBILITY TRADE-OFF & EXPERIMENTAL VALIDATION")
    print("=" * 75)

    # 1. Load model bundle
    model_path = os.path.join(models_dir, "best_coating_model.joblib")
    bundle = load_model_bundle(model_path)
    print(f"Loaded trained model: {bundle['model_name']}")

    # 2. Hardness-Flexibility Trade-off Analysis
    print("\n[Part 1] Generating Feasible Design Space & Computing Pareto Frontier...")
    candidates_df = generate_tradeoff_candidates(bundle, n_samples=350, random_state=42)
    pareto_df = compute_pareto_frontier(candidates_df, x_col="Flexibility", y_col="Hardness")

    print(f"  - Total Feasible Candidate Formulations : {len(candidates_df)}")
    print(f"  - Non-Dominated Pareto Optimal Solutions: {len(pareto_df)}")

    # Selected candidate from Stage 5 inverse design (or representative recipe)
    selected_recipe = {
        "Resin_Content": 52.78,
        "Pigment_Content": 22.32,
        "Solvent_Content": 16.80,
        "Additive_Content": 1.19,
        "Curing_Agent_Content": 6.91,
        "Curing_Temperature": 93.2,
        "Curing_Time": 59.3
    }
    pred_selected = predict_properties(selected_recipe, model_bundle=bundle)
    pred_dict = dict(zip(pred_selected["Property"], pred_selected["Predicted Value"]))

    highlighted_coords = {
        "Flexibility": pred_dict["Flexibility"],
        "Hardness": pred_dict["Hardness"]
    }

    tradeoff_plot_path = os.path.join(plots_dir, "hardness_flexibility_tradeoff.png")
    plot_hardness_flexibility_tradeoff(
        candidates_df,
        pareto_df,
        highlighted_point=highlighted_coords,
        save_path=tradeoff_plot_path
    )

    # Save Pareto points to report
    pareto_csv_path = os.path.join(reports_dir, "pareto_optimal_formulations.csv")
    pareto_df.to_csv(pareto_csv_path, index=False)
    print(f"[Saved] Pareto optimal formulations -> {pareto_csv_path}")

    # 3. Experimental Bench Validation
    print("\n[Part 2] Benchmarking AI Predictions Against Experimental Measurements...")
    exp_measured = {
        "Hardness": args.exp_hardness,
        "Adhesion": args.exp_adhesion,
        "Flexibility": args.exp_flexibility,
        "Gloss": args.exp_gloss,
        "Drying_Time": args.exp_drying_time
    }

    val_df, overall_mae, overall_mape = evaluate_experimental_validation(pred_dict, exp_measured)

    print("\n" + "=" * 75)
    print("AI PREDICTION VS. LABORATORY EXPERIMENTAL MEASUREMENTS")
    print("=" * 75)
    print(val_df.to_string(index=False))
    print("-" * 75)
    print(f"Overall Validation Mean Absolute Error (MAE)       : {overall_mae:.2f}")
    print(f"Overall Validation Mean Absolute Percentage Error  : {overall_mape:.2f}%")
    print("=" * 75)

    # Save experimental validation report and plot
    val_csv_path = os.path.join(reports_dir, "experimental_validation_report.csv")
    val_df.to_csv(val_csv_path, index=False)
    print(f"[Saved] Experimental validation report -> {val_csv_path}")

    val_plot_path = os.path.join(plots_dir, "experimental_validation_comparison.png")
    plot_ai_vs_experimental_comparison(val_df, overall_mae, overall_mape, save_path=val_plot_path)

    print("\n" + "=" * 75)
    print("[SUCCESS] STAGE 6 COMPLETE: Trade-off & Experimental Validation Finished.")
    print("=" * 75)


if __name__ == "__main__":
    main()
