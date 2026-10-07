"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Script: run_inverse_design.py
Description: CLI demonstration for Stage 5 inverse design: searches for the
             optimal coating recipe and baking conditions to hit desired targets.
=============================================================================
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.optimization import inverse_design, print_inverse_design_results


def main():
    parser = argparse.ArgumentParser(
        description="Inverse formulation optimization using Differential Evolution."
    )
    # Target property desired values
    parser.add_argument("--hardness", type=float, default=82.0, help="Target Hardness (default: 82.0)")
    parser.add_argument("--adhesion", type=float, default=78.0, help="Target Adhesion (default: 78.0)")
    parser.add_argument("--flexibility", type=float, default=45.0, help="Target Flexibility (default: 45.0)")
    parser.add_argument("--gloss", type=float, default=80.0, help="Target Gloss (default: 80.0)")
    parser.add_argument("--drying_time", type=float, default=50.0, help="Target Drying Time (default: 50.0)")

    # Weights
    parser.add_argument("--w_hardness", type=float, default=1.0)
    parser.add_argument("--w_adhesion", type=float, default=1.0)
    parser.add_argument("--w_flexibility", type=float, default=1.0)
    parser.add_argument("--w_gloss", type=float, default=1.0)
    parser.add_argument("--w_drying", type=float, default=1.0)

    # Optimization parameters
    parser.add_argument("--popsize", type=int, default=25, help="Population size factor (default: 25)")
    parser.add_argument("--maxiter", type=int, default=250, help="Max evolutionary generations (default: 250)")

    args = parser.parse_args()

    targets = {}
    weights = {}

    if args.hardness is not None:
        targets["Hardness"] = args.hardness
        weights["Hardness"] = args.w_hardness
    if args.adhesion is not None:
        targets["Adhesion"] = args.adhesion
        weights["Adhesion"] = args.w_adhesion
    if args.flexibility is not None:
        targets["Flexibility"] = args.flexibility
        weights["Flexibility"] = args.w_flexibility
    if args.gloss is not None:
        targets["Gloss"] = args.gloss
        weights["Gloss"] = args.w_gloss
    if args.drying_time is not None:
        targets["Drying_Time"] = args.drying_time
        weights["Drying_Time"] = args.w_drying

    print("=" * 75)
    print("STAGE 5: INVERSE COATING FORMULATION OPTIMIZATION")
    print("=" * 75)
    print("Target Specifications:")
    for tgt, val in targets.items():
        print(f"  - {tgt:<15}: {val:>6.2f} (Weight: {weights[tgt]:.1f})")

    print("\nExecuting Differential Evolution global stochastic search...")
    results = inverse_design(
        target_properties=targets,
        property_weights=weights,
        popsize=args.popsize,
        maxiter=args.maxiter,
        random_state=42
    )

    # Print formatted output
    print_inverse_design_results(results)

    # Save to report CSV
    reports_dir = os.path.join(SCRIPT_DIR, "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_csv = os.path.join(reports_dir, "inverse_design_results.csv")
    results["comparison_table"].to_csv(report_csv, index=False)
    print(f"\n[Saved] Inverse design comparison table saved to: {report_csv}")


if __name__ == "__main__":
    main()
