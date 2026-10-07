"""
=============================================================================
AI-Driven Coating Formulation & Property Optimization
Module: src/visualization.py
Description: Publication-quality plotting and visualization routines for
             Exploratory Data Analysis (EDA), model evaluation, trade-off
             curves, and validation diagnostics.
=============================================================================
"""

import os
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set engineering visual theme
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True
})

FEATURE_UNITS = {
    "Resin_Content": "wt%",
    "Pigment_Content": "wt%",
    "Solvent_Content": "wt%",
    "Additive_Content": "wt%",
    "Curing_Agent_Content": "wt%",
    "Curing_Temperature": "°C",
    "Curing_Time": "minutes",
    "Hardness": "Index (0-100)",
    "Adhesion": "Index (0-100)",
    "Flexibility": "Index (0-100)",
    "Gloss": "GU @ 60°",
    "Drying_Time": "minutes"
}


def plot_feature_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot univariate distribution histograms with kernel density estimation (KDE).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to plot.
    columns : Optional[List[str]]
        Specific columns to plot. If None, plots all numeric columns.
    save_path : Optional[str]
        File path to save the generated figure.
    """
    if columns is None:
        columns = list(FEATURE_UNITS.keys())

    # 4 columns per row layout
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columns):
        ax = axes[i]
        unit = FEATURE_UNITS.get(col, "")
        sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4", edgecolor="black", alpha=0.6)
        ax.set_title(f"Distribution of {col.replace('_', ' ')}", fontweight="bold")
        ax.set_xlabel(f"{col.replace('_', ' ')} [{unit}]")
        ax.set_ylabel("Count / Frequency")

    # Hide extra unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Univariate Distributions of Coating Formulation Inputs & Target Properties", fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Feature distributions -> {save_path}")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Pearson correlation matrix heatmap for inputs and targets.
    """
    if columns is None:
        columns = [c for c in FEATURE_UNITS.keys() if c in df.columns]

    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        ax=ax
    )

    ax.set_title("Correlation Heatmap: Formulation Inputs vs. Coating Target Properties", fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Correlation matrix -> {save_path}")

    return fig


def plot_coating_domain_relationships(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot key polymer/coating science domain relationships:
    1. Curing Agent Content vs. Hardness (Crosslink density effect)
    2. Resin Content vs. Flexibility (Polymer chain ductility)
    3. Pigment Content vs. Gloss (Micro-roughness scattering)
    4. Curing Temperature vs. Drying Time (Kinetic reaction acceleration)
    5. Curing Temperature vs. Adhesion (Optimal cure vs overcure stress)
    6. Hardness vs. Flexibility (The classic Coating Trade-Off)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Curing Agent vs Hardness
    sns.regplot(
        data=df, x="Curing_Agent_Content", y="Hardness", ax=axes[0, 0],
        scatter_kws={"alpha": 0.4, "color": "#1f77b4"}, line_kws={"color": "#d62728"}
    )
    axes[0, 0].set_title("1. Crosslinker Effect on Hardness", fontweight="bold")
    axes[0, 0].set_xlabel("Curing Agent Content [wt%]")
    axes[0, 0].set_ylabel("Hardness [0-100 Index]")

    # 2. Resin vs Flexibility
    sns.regplot(
        data=df, x="Resin_Content", y="Flexibility", ax=axes[0, 1],
        scatter_kws={"alpha": 0.4, "color": "#2ca02c"}, line_kws={"color": "#d62728"}
    )
    axes[0, 1].set_title("2. Polymer Binder Effect on Flexibility", fontweight="bold")
    axes[0, 1].set_xlabel("Resin Content [wt%]")
    axes[0, 1].set_ylabel("Flexibility [0-100 Index]")

    # 3. Pigment vs Gloss
    sns.regplot(
        data=df, x="Pigment_Content", y="Gloss", ax=axes[0, 2],
        scatter_kws={"alpha": 0.4, "color": "#ff7f0e"}, line_kws={"color": "#d62728"}
    )
    axes[0, 2].set_title("3. Pigment Matting Effect on 60° Gloss", fontweight="bold")
    axes[0, 2].set_xlabel("Pigment Content [wt%]")
    axes[0, 2].set_ylabel("Specular Gloss [GU @ 60°]")

    # 4. Temperature vs Drying Time
    sns.regplot(
        data=df, x="Curing_Temperature", y="Drying_Time", ax=axes[1, 0],
        scatter_kws={"alpha": 0.4, "color": "#9467bd"}, line_kws={"color": "#d62728"}
    )
    axes[1, 0].set_title("4. Thermal Acceleration of Drying Rate", fontweight="bold")
    axes[1, 0].set_xlabel("Curing Temperature [°C]")
    axes[1, 0].set_ylabel("Drying Time [minutes]")

    # 5. Temperature vs Adhesion
    sns.scatterplot(
        data=df, x="Curing_Temperature", y="Adhesion", ax=axes[1, 1],
        hue="Curing_Agent_Content", palette="viridis", alpha=0.7
    )
    axes[1, 1].set_title("5. Curing Temperature vs. Substrate Adhesion", fontweight="bold")
    axes[1, 1].set_xlabel("Curing Temperature [°C]")
    axes[1, 1].set_ylabel("Adhesion [0-100 Index]")

    # 6. Hardness vs Flexibility Trade-off
    sns.scatterplot(
        data=df, x="Flexibility", y="Hardness", ax=axes[1, 2],
        hue="Curing_Agent_Content", palette="magma", alpha=0.75
    )
    axes[1, 2].set_title("6. Hardness vs. Flexibility Trade-off", fontweight="bold")
    axes[1, 2].set_xlabel("Flexibility [0-100 Index]")
    axes[1, 2].set_ylabel("Hardness [0-100 Index]")

    fig.suptitle("Core Physical & Chemical Correlations in Coating Formulation", fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Domain relationships -> {save_path}")

    return fig


def export_eda_summary_statistics(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate comprehensive numerical statistics (mean, std, min, Q25, median, Q75, max, skewness).
    """
    stats_df = df.describe().T
    stats_df["skew"] = df.skew(numeric_only=True)
    stats_df["unit"] = [FEATURE_UNITS.get(idx, "") for idx in stats_df.index]
    
    # Reorder columns
    stats_df = stats_df[["unit", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew"]]
    stats_df = stats_df.round(3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_df.to_csv(save_path)
        print(f"[Report Saved] EDA summary statistics -> {save_path}")

    return stats_df


def plot_hardness_flexibility_tradeoff(
    candidates_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    highlighted_point: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the Hardness vs. Flexibility Trade-off with Pareto Frontier.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Feasible candidate formulations with predicted Hardness and Flexibility.
    pareto_df : pd.DataFrame
        Pareto-optimal non-dominated solutions.
    highlighted_point : Optional[Dict[str, float]]
        Coordinates (Flexibility, Hardness) of the user's optimized formulation.
    save_path : Optional[str]
        File path to save the generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Feasible design space scatter
    scatter = ax.scatter(
        candidates_df["Flexibility"],
        candidates_df["Hardness"],
        c=candidates_df.get("Curing_Agent_Content", "#1f77b4"),
        cmap="coolwarm",
        alpha=0.6,
        edgecolor="k",
        linewidth=0.5,
        s=55,
        label="Feasible Formulations"
    )

    if "Curing_Agent_Content" in candidates_df.columns:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Curing Agent Content [wt%]", rotation=270, labelpad=18)

    # 2. Pareto Frontier
    sorted_pareto = pareto_df.sort_values(by="Flexibility")
    ax.plot(
        sorted_pareto["Flexibility"],
        sorted_pareto["Hardness"],
        color="#d62728",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Pareto Frontier (Non-Dominated Boundary)"
    )

    # 3. Highlighted Point (e.g. Optimized inverse design recommendation)
    if highlighted_point is not None:
        h_flex = highlighted_point.get("Flexibility")
        h_hard = highlighted_point.get("Hardness")
        if h_flex is not None and h_hard is not None:
            ax.scatter(
                [h_flex], [h_hard],
                color="#ffdd00",
                edgecolor="black",
                linewidth=2.0,
                s=240,
                marker="*",
                zorder=10,
                label=f"Selected Formulation (H: {h_hard:.1f}, F: {h_flex:.1f})"
            )
            ax.annotate(
                f"Selected\n(H:{h_hard:.1f}, F:{h_flex:.1f})",
                xy=(h_flex, h_hard),
                xytext=(h_flex + 2.5, h_hard - 4.0),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=7),
                fontweight="bold",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="k", lw=0.8)
            )

    ax.set_title("Hardness vs. Flexibility Trade-off & Pareto Optimal Frontier", fontweight="bold", fontsize=14, pad=12)
    ax.set_xlabel("Flexibility [0-100 Index] (Higher = More Ductile / Impact Resistant)", fontsize=12)
    ax.set_ylabel("Hardness [0-100 Index] (Higher = Superior Scratch Resistance)", fontsize=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Hardness-Flexibility trade-off -> {save_path}")

    return fig


def plot_ai_vs_experimental_comparison(
    val_df: pd.DataFrame,
    mae_val: float,
    mape_val: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate grouped bar chart comparing AI-predicted properties against
    actual bench experimental measurements.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    properties = val_df["Property"].tolist()
    pred_vals = val_df["AI-Predicted"].tolist()
    exp_vals = val_df["Experimental"].tolist()

    x = np.arange(len(properties))
    width = 0.35

    rects1 = ax.bar(x - width/2, pred_vals, width, label="AI Predicted", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, exp_vals, width, label="Lab Experimental", color="#2ca02c", edgecolor="black", alpha=0.85)

    # Annotate bars with values
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Measured Property Magnitude", fontsize=12)
    ax.set_title(
        f"Experimental Validation: AI Predicted vs. Laboratory Measurements\n(Overall MAE: {mae_val:.2f} | MAPE: {mape_val:.2f}%)",
        fontweight="bold",
        fontsize=13,
        pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(properties, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.set_ylim(0, max(max(pred_vals), max(exp_vals)) * 1.18)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot Saved] Experimental validation -> {save_path}")

    return fig

