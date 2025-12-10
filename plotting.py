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


# ---------------------------------------------------------
# FIGURE 4
# ---------------------------------------------------------

def plot_figure3(mean_M: pd.DataFrame, mean_H: dict, task_name="Task"):
    """
    Reproduce Figure 3 from the paper:
    Distribution of Δ = μM(x) - μH(x) across instances.
    """

    probes = ["Lexical variability", "Syntactic variability", "Semantic variability"]
    probe_cols = ["M_lexical", "M_syntactic", "M_semantic"]

    # For each probe, compute Δ across strategies
    deltas = []
    for _, row in mean_M.iterrows():
        deltas.append([
            row["M_lexical"] - mean_H["H_lexical"],
            row["M_syntactic"] - mean_H["H_syntactic"],
            row["M_semantic"] - mean_H["H_semantic"]
        ])

    deltas = np.array(deltas)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for i, ax in enumerate(axes):
        ax.boxplot(deltas[:, i], vert=False, patch_artist=True,
                   boxprops=dict(facecolor="orange", alpha=0.7))
        ax.set_title(f"{probes[i]} Δ = μM - μH")
        ax.set_xlabel("Distance difference")

    plt.suptitle(f"Figure 3 — {task_name}")
    plt.tight_layout()
    plt.show()


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

    probe_names = ["Cosine distance", "POS bigram distance", "Unigram distance"]
    probe_cols  = ["W_semantic", "W_syntactic", "W_lexical"]

    colors = {
        "human": "red",
        "ancestral": "blue",
        "top_k": "green",
        "top_p": "orange",
        "temperature": "purple",
    }

    plt.figure(figsize=(10, 5))

    for _, row in mean_W_with_human.iterrows():
        strat = row["strategy"]
        color = colors.get(strat, "gray")
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
