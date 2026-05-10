"""
visualize.py
============
Generates all figures for the Jupyter notebook and README.
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

ACCENT   = "#20808D"
ACCENT2  = "#A84B2F"
DARK     = "#1B474D"
LIGHT    = "#BCE2E7"
GOLD     = "#FFC553"
GRAY     = "#7A7974"
BG       = "#FAFAF8"

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.6,
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

with open("cache/results.pkl", "rb") as f:
    R = pickle.load(f)

ff5_factors  = R["ff5_factors"]
returns      = R["returns"]
hm           = R["hm"]
ff5_summary  = R["ff5_summary"]
ff5hm_summary= R["ff5hm_summary"]
ff5_gammas   = R["ff5_gammas"]
ff5hm_gammas = R["ff5hm_gammas"]
ff5_betas    = R["ff5_betas"]
ff5hm_betas  = R["ff5hm_betas"]
grs5         = R["ff5_grs"]
grs6         = R["ff5hm_grs"]
r2_imp       = R["r2_improvement"]
avg_r2_ff5   = R["avg_cs_r2_ff5"]
avg_r2_ff5hm = R["avg_cs_r2_ff5hm"]


# ── Figure 1: Factor Premium Bar Chart ───────────────────────────────────────
def fig_factor_premia():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG)

    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "HM"]
    labels  = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "HM\n(Hiring)"]

    means   = [ff5hm_summary.loc[f, "mean"]   * 100 if f in ff5hm_summary.index else 0 for f in factors]
    tstats  = [ff5hm_summary.loc[f, "t_stat"] if f in ff5hm_summary.index else 0 for f in factors]
    sig     = [ff5hm_summary.loc[f, "significant"] if f in ff5hm_summary.index else False for f in factors]

    colors = [ACCENT if s else GRAY for s in sig]
    colors[-1] = ACCENT2  # HM always highlighted

    bars = ax.bar(labels, means, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    for bar, t, m in zip(bars, tstats, means):
        ypos = bar.get_height() + 0.01 if m >= 0 else bar.get_height() - 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"t={t:.2f}", ha="center", va="bottom", fontsize=9, color="#444")

    ax.axhline(0, color="#333", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax.set_title("Fama-MacBeth Factor Premia — FF5 + Hiring Momentum\n(Newey-West Corrected, 2021–2025)")
    ax.set_ylabel("Avg Monthly Premium (×100 = %)")
    ax.set_xlabel("")

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT,  label="Significant (p<0.05)"),
        Patch(facecolor=GRAY,    label="Not significant"),
        Patch(facecolor=ACCENT2, label="Hiring Momentum (HM)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0)

    fig.tight_layout()
    fig.savefig("figures/factor_premia.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/factor_premia.png")


# ── Figure 2: Cross-Sectional R² Over Time ───────────────────────────────────
def fig_cs_r2():
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)

    r2_ff5   = ff5_gammas["cs_r2"].rolling(6).mean()
    r2_ff5hm = ff5hm_gammas["cs_r2"].rolling(6).mean()

    ax.plot(r2_ff5.index,   r2_ff5.values,   color=GRAY,   linewidth=2,   label="FF5", alpha=0.85)
    ax.plot(r2_ff5hm.index, r2_ff5hm.values, color=ACCENT, linewidth=2.2, label="FF5 + HM (Hiring)", alpha=0.95)
    ax.fill_between(r2_ff5hm.index, r2_ff5.values, r2_ff5hm.values,
                    alpha=0.15, color=ACCENT, label=f"R² lift ({r2_imp:.1f}% avg)")

    avg5  = ff5_gammas["cs_r2"].mean()
    avg6  = ff5hm_gammas["cs_r2"].mean()
    ax.axhline(avg5,  color=GRAY,   linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(avg6,  color=ACCENT, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title("Cross-Sectional R² Over Time\nFF5 vs. FF5 + Hiring Momentum (6-month rolling avg)")
    ax.set_ylabel("Cross-Sectional R²")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0)
    fig.tight_layout()
    fig.savefig("figures/cs_r2_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/cs_r2_over_time.png")


# ── Figure 3: Factor Correlation Heatmap ─────────────────────────────────────
def fig_factor_corr():
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    common_idx = ff5_factors.index.intersection(hm.index)
    combined = pd.concat([ff5_factors[factor_cols].reindex(common_idx),
                          hm.reindex(common_idx).rename("HM (Hiring)")], axis=1)
    corr = combined.corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(BG)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0, linewidths=0.5,
                linecolor="white", square=True, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10})

    ax.set_title("Factor Correlation Matrix\nFF5 + Hiring Momentum (2018–2025)")
    fig.tight_layout()
    fig.savefig("figures/factor_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/factor_correlation.png")


# ── Figure 4: Rolling Gamma (HM premium over time) ───────────────────────────
def fig_rolling_hm_gamma():
    if "HM" not in ff5hm_gammas.columns:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor(BG)

    hm_gamma = ff5hm_gammas["HM"]
    rolling_mean = hm_gamma.rolling(6).mean()
    cumulative   = hm_gamma.cumsum()

    ax1 = axes[0]
    ax1.bar(hm_gamma.index, hm_gamma.values * 100,
            color=[ACCENT if v >= 0 else ACCENT2 for v in hm_gamma.values],
            alpha=0.7, width=20)
    ax1.plot(rolling_mean.index, rolling_mean.values * 100,
             color=DARK, linewidth=2, label="6-month rolling avg")
    ax1.axhline(0, color="#333", linewidth=0.8)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax1.set_title("Hiring Momentum (HM) Cross-Sectional Risk Premium Over Time")
    ax1.set_ylabel("Monthly Premium (%)")
    ax1.legend(fontsize=9, framealpha=0)

    ax2 = axes[1]
    ax2.plot(cumulative.index, cumulative.values * 100,
             color=ACCENT, linewidth=2.2)
    ax2.fill_between(cumulative.index, 0, cumulative.values * 100,
                     alpha=0.15, color=ACCENT)
    ax2.axhline(0, color="#333", linewidth=0.8)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.set_title("Cumulative HM Factor Premium")
    ax2.set_ylabel("Cumulative Premium (%)")

    fig.tight_layout()
    fig.savefig("figures/hm_gamma_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/hm_gamma_over_time.png")


# ── Figure 5: Model Comparison Summary ───────────────────────────────────────
def fig_model_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("FF5 vs. FF5 + Hiring Momentum — Model Comparison", fontsize=13, fontweight="bold")

    # (a) Avg CS R2
    ax = axes[0]
    vals = [avg_r2_ff5 * 100, avg_r2_ff5hm * 100]
    bars = ax.bar(["FF5", "FF5+HM"], vals,
                  color=[GRAY, ACCENT], width=0.45, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Avg Cross-Sectional R²")
    ax.set_ylabel("R²  (%)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # (b) Mean absolute alpha
    mean_alpha_ff5   = ff5_betas["alpha"].abs().mean() * 100
    mean_alpha_ff5hm = ff5hm_betas["alpha"].abs().mean() * 100
    ax = axes[1]
    vals2 = [mean_alpha_ff5, mean_alpha_ff5hm]
    bars2 = ax.bar(["FF5", "FF5+HM"], vals2,
                   color=[GRAY, ACCENT], width=0.45, edgecolor="white")
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.3f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Mean |Alpha| (monthly)")
    ax.set_ylabel("Mean |Alpha| (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))

    # (c) GRS test stat
    ax = axes[2]
    grs_stats = [grs5["grs_stat"], grs6["grs_stat"]]
    if not any(np.isnan(grs_stats)):
        bars3 = ax.bar(["FF5", "FF5+HM"], grs_stats,
                       color=[GRAY, ACCENT], width=0.45, edgecolor="white")
        for bar, v in zip(bars3, grs_stats):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("GRS Test Statistic\n(lower = better model fit)")
        ax.set_ylabel("GRS F-statistic")
    else:
        ax.text(0.5, 0.5, "GRS N/A\n(insufficient portfolios)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("GRS Test Statistic")

    fig.tight_layout()
    fig.savefig("figures/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/model_comparison.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_factor_premia()
    fig_cs_r2()
    fig_factor_corr()
    fig_rolling_hm_gamma()
    fig_model_comparison()
    print("All figures saved to figures/")
