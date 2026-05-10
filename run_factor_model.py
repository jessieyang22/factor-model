#!/usr/bin/env python3
"""
run_factor_model.py
===================
Production entry point for the Fama-French 5-Factor + Hiring Momentum analysis.

Usage:
    python run_factor_model.py                  # full run (downloads data if not cached)
    python run_factor_model.py --no-cache       # force re-download all data
    python run_factor_model.py --no-plots       # skip figure generation
    python run_factor_model.py --summary-only   # print results, skip plots and cache
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="FF5 + Hiring Momentum Factor Model")
    parser.add_argument("--no-cache",      action="store_true", help="Force re-download data")
    parser.add_argument("--no-plots",      action="store_true", help="Skip figure generation")
    parser.add_argument("--summary-only",  action="store_true", help="Print summary and exit")
    args = parser.parse_args()

    os.makedirs("cache",   exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    if args.no_cache:
        for f in ["cache/prices.parquet", "cache/ff5.zip"]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed cached {f}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n── Step 1/4: Loading data ─────────────────────────────────────")
    from data_loader import load_ff5_factors, load_stock_returns, load_hiring_momentum_factor
    ff5     = load_ff5_factors()
    returns = load_stock_returns()
    hm      = load_hiring_momentum_factor(ff5.index)

    print(f"\n  Universe:      {returns.shape[1]} stocks")
    print(f"  Sample:        {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Observations:  {returns.shape[0]} months")

    if args.summary_only:
        print("\n[--summary-only] Skipping model estimation. Run without flag for full output.")
        return

    # ── Run model ──────────────────────────────────────────────────────────
    print("\n── Step 2/4: Running Fama-MacBeth estimation ─────────────────")
    from factor_model import compare_models
    comparison = compare_models(returns, ff5, hm)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n── Step 3/4: Results ─────────────────────────────────────────")
    r2_ff5   = comparison["ff5"]["fm"]["avg_cs_r2"]
    r2_ff5hm = comparison["ff5_hm"]["fm"]["avg_cs_r2"]
    r2_imp   = comparison["r2_improvement_pct"]
    grs5     = comparison["ff5"]["grs"]
    grs6     = comparison["ff5_hm"]["grs"]

    print("\nFF5 Fama-MacBeth (Newey-West, lag=4):")
    print(comparison["ff5"]["fm"]["summary"].round(4).to_string())

    print("\nFF5+HM Fama-MacBeth (Newey-West, lag=4):")
    print(comparison["ff5_hm"]["fm"]["summary"].round(4).to_string())

    print(f"\nCross-Sectional R²:  FF5={r2_ff5:.3f}  |  FF5+HM={r2_ff5hm:.3f}  |  Lift={r2_imp:+.1f}%")
    print(f"GRS (FF5):    F={grs5['grs_stat']:.3f}  p={grs5['p_value']:.4f}  Reject={grs5['reject_H0']}")
    print(f"GRS (FF5+HM): F={grs6['grs_stat']:.3f}  p={grs6['p_value']:.4f}  Reject={grs6['reject_H0']}")

    # ── Save results ───────────────────────────────────────────────────────
    summary_ff5   = comparison["ff5"]["fm"]["summary"]
    summary_ff5hm = comparison["ff5_hm"]["fm"]["summary"]
    out = pd.concat([summary_ff5.add_suffix("_FF5"),
                     summary_ff5hm.add_suffix("_FF5HM")], axis=1)
    out.to_csv("results_fama_macbeth.csv")
    print("\n  Saved results to results_fama_macbeth.csv")

    # ── Generate figures ───────────────────────────────────────────────────
    if not args.no_plots:
        print("\n── Step 4/4: Generating figures ──────────────────────────────")
        import pickle
        with open("cache/results.pkl", "wb") as f:
            pickle.dump({
                "ff5_summary":    summary_ff5,
                "ff5hm_summary":  summary_ff5hm,
                "ff5_gammas":     comparison["ff5"]["fm"]["gammas"],
                "ff5hm_gammas":   comparison["ff5_hm"]["fm"]["gammas"],
                "ff5_betas":      comparison["ff5"]["betas"],
                "ff5hm_betas":    comparison["ff5_hm"]["betas"],
                "ff5_grs":        grs5,
                "ff5hm_grs":      grs6,
                "r2_improvement": r2_imp,
                "hm_significant": comparison["hm_significant"],
                "avg_cs_r2_ff5":  r2_ff5,
                "avg_cs_r2_ff5hm":r2_ff5hm,
                "ff5_factors":    ff5,
                "returns":        returns,
                "hm":             hm,
            }, f)
        from visualize import (fig_factor_premia, fig_cs_r2, fig_factor_corr,
                                fig_rolling_hm_gamma, fig_model_comparison)
        fig_factor_premia()
        fig_cs_r2()
        fig_factor_corr()
        fig_rolling_hm_gamma()
        fig_model_comparison()
        print("  All figures saved to figures/")

    print("\n── Complete ──────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
