"""
plotting.py

Helper functions to reproduce Figures 4, 5, 6 from the paper:
- Figure 4: Mean difference Δ = E[M] - E[H]
- Figure 5: Mean Wasserstein distance W(C, H) by strategy + human baseline
- Figure 6: Distribution of W(C, H) across examples

Usage:
    import plotting
    plotting.plot_figure5(mean_W_with_human)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import seaborn as sns


# ---------------------------------------------------------
# FIGURE 5
# ---------------------------------------------------------

def plot_figure5(mean_W_with_human: pd.DataFrame, task_name="Task"):
    """
    Reproduce simplified version of Figure 5:
    x = mean W(C,H)
    y = probe type
    color = strategy (human in red)

    Input columns required:
    strategy | W_lexical | W_syntactic | W_semantic
    """

    def strategy_to_color(strategy: str) -> str:
        if strategy == "human":
            return "red"
        if strategy == "ancestral":
            return "blue"
        if strategy.startswith("top_k"):
            return "gray"
        if strategy.startswith("top_p"):
            return "orange"
        if strategy.startswith("temp"):
            return "purple"
        if strategy.startswith("locally_typical"):
            return "green"
        return "gray"

    probe_names = ["Cosine distance", "POS bigram distance", "Unigram distance"]
    probe_cols  = ["W_semantic", "W_syntactic", "W_lexical"]

    plt.figure(figsize=(10, 5))

    for _, row in mean_W_with_human.iterrows():
        strat = row["strategy"]
        color = strategy_to_color(strat)
        for i, col in enumerate(probe_cols):
            x = row[col]
            plt.scatter(x, i, s=80, color=color, label=strat)

    plt.yticks([0, 1, 2], probe_names)
    plt.xlabel("Mean W₁(C(x), H(x))")
    plt.title(f"Task = {task_name}")

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    plt.legend(
        legend_dict.values(),
        legend_dict.keys(),
        title="Decoder",
        bbox_to_anchor=(1.05, 1)
    )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# FIGURE 6
# ---------------------------------------------------------

def plot_figure6(all_W: list, task_name="Task"):
    """
    Plot Figure 6-like distributions:
    Show histograms of W(C,H) across examples for the 3 probes.
    """

    df_W = pd.DataFrame(all_W)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharey=False)

    probes = [
        ("W_semantic", "Cosine distance"),
        ("W_syntactic", "POS bigram distance"),
        ("W_lexical", "Unigram distance"),
    ]

    for ax, (col, title) in zip(axs, probes):
        ax.hist(df_W[col], bins=30, color="steelblue", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("W₁(C(x), H(x))")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Task = {task_name}: Distribution of Wasserstein Distances")
    plt.tight_layout()
    plt.show()


def plot_figure3_from_deltas(df, task_name="Task", label="μM(x) − μH(x)"):
    probes = ["Lexical variability", "Syntactic variability", "Semantic variability"]
    cols = ["delta_lexical", "delta_syntactic", "delta_semantic"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for i, ax in enumerate(axes):
        ax.boxplot(
            df[cols[i]],
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="orange", alpha=0.7),
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(0.9,1.1)
        ax.set_title(probes[i])
        ax.set_xlabel(label)

    plt.suptitle(f"{task_name} Variability Across Probes")
    plt.tight_layout()
    plt.show()



def plot_human_vs_model_histograms(
    df_H,
    df_M,
    *,
    task_title="Text Simplification: Human vs Model Distance Comparison",
    panel_title_prefix="Figure 3 —",
    exclude_strategies=("locally_typical_0.2", "top_p_0.95"),
    bins=40,
    xlim=(0, 1),
    figsize=(15, 4),
    human_alpha=1.0,
    model_alpha=0.1,
    human_lw=2.8,
    model_lw=1.5,
    style="whitegrid",
    parse_lists=True,
    list_cols_H=("H_lexical", "H_syntactic", "H_semantic"),
    list_cols_M=("M_lexical", "M_syntactic", "M_semantic"),
    show_legend=True,
):
    """
    Plots per-probe histograms comparing Human (H) variability vs Model (M) variability by strategy.

    Inputs:
      df_H: DataFrame with list columns H_lexical/H_syntactic/H_semantic (strings or lists)
      df_M: DataFrame with list columns M_lexical/M_syntactic/M_semantic + 'strategy' column

    Notes:
      - If parse_lists=True, will ast.literal_eval the list columns when they are strings.
      - Returns (fig, axes).
    """
    # Defensive copy (avoid mutating caller dfs)
    df_H = df_H.copy()
    df_M = df_M.copy()

    # Parse list columns if needed
    if parse_lists:
        for c in list_cols_H:
            if c in df_H.columns and len(df_H) and isinstance(df_H[c].iloc[0], str):
                df_H[c] = df_H[c].apply(ast.literal_eval)
        for c in list_cols_M:
            if c in df_M.columns and len(df_M) and isinstance(df_M[c].iloc[0], str):
                df_M[c] = df_M[c].apply(ast.literal_eval)

    # Build pooled human arrays
    all_H = {
        "lexical":   np.concatenate(df_H["H_lexical"].values),
        "syntactic": np.concatenate(df_H["H_syntactic"].values),
        "semantic":  np.concatenate(df_H["H_semantic"].values),
    }

    # Build pooled model arrays per strategy
    all_M = {}
    for strategy, g in df_M.groupby("strategy"):
        all_M[strategy] = {
            "lexical":   np.concatenate(g["M_lexical"].values),
            "syntactic": np.concatenate(g["M_syntactic"].values),
            "semantic":  np.concatenate(g["M_semantic"].values),
        }

    sns.set_style(style)

    probes = [
        ("lexical",   "Lexical variability"),
        ("syntactic", "Syntactic variability"),
        ("semantic",  "Semantic variability"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, (probe, title) in zip(axes, probes):
        # Human
        sns.histplot(
            all_H[probe],
            bins=bins,
            stat="probability",
            element="step",
            linewidth=human_lw,
            alpha=human_alpha,
            color="black",
            label="Human",
            ax=ax,
        )

        # Model strategies
        for strategy, vals in all_M.items():
            if strategy in exclude_strategies:
                continue
            sns.histplot(
                vals[probe],
                bins=bins,
                stat="probability",
                element="step",
                linewidth=model_lw,
                alpha=model_alpha,
                label=strategy,
                ax=ax,
            )

        ax.set_title(title)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Probability")
        ax.set_xlim(*xlim)

    # Figure title + panel labels
    fig.suptitle(task_title, fontsize=14, y=1.05)
    fig.text(0.17, -0.02, "(a) Lexical variability", ha="center", fontsize=11)
    fig.text(0.50, -0.02, "(b) Syntactic variability", ha="center", fontsize=11)
    fig.text(0.83, -0.02, "(c) Semantic variability", ha="center", fontsize=11)

    if show_legend:
        axes[0].legend(fontsize=8)

    plt.tight_layout()
    return fig, axes

