"""
factor_model.py
===============
Core analytics:
  - Fama-MacBeth cross-sectional regression
  - Newey-West corrected standard errors
  - GRS test for joint alpha significance
  - Factor contribution analysis (FF5 vs FF5+HM)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


# ── Time-Series Betas ────────────────────────────────────────────────────────

def estimate_ts_betas(returns: pd.DataFrame, factors: pd.DataFrame,
                      min_obs: int = 24) -> pd.DataFrame:
    """
    For each stock, run a time-series regression of excess returns on factors.
    Returns DataFrame of betas (stocks x factors) and alphas.
    """
    results = {}
    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0, index=factors.index)
    factor_cols = [c for c in factors.columns if c != "RF"]
    X = sm.add_constant(factors[factor_cols])

    for ticker in returns.columns:
        y = returns[ticker] - rf
        aligned = pd.concat([y, X], axis=1).dropna()
        if len(aligned) < min_obs:
            continue
        try:
            model = sm.OLS(aligned.iloc[:, 0], aligned.iloc[:, 1:]).fit()
            row = {"alpha": model.params["const"]}
            row.update({f: model.params[f] for f in factor_cols if f in model.params})
            row["r2"] = model.rsquared
            row["alpha_tstat"] = model.tvalues["const"]
            results[ticker] = row
        except Exception:
            continue

    betas = pd.DataFrame(results).T
    print(f"  Time-series betas estimated for {len(betas)} stocks")
    return betas


# ── Fama-MacBeth Regression ──────────────────────────────────────────────────

def fama_macbeth(returns: pd.DataFrame, factors: pd.DataFrame,
                 lag: int = 4) -> dict:
    """
    Fama-MacBeth (1973) two-pass procedure:
      Pass 1: For each stock, regress excess returns on factors (rolling 36-month window)
              to get time-varying betas.
      Pass 2: Each month, cross-sectional regression of returns on betas.
              Average the cross-sectional slope estimates; apply Newey-West SEs.

    lag: Newey-West lag order (default 4 for monthly data)
    """
    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0, index=factors.index)
    factor_cols = [c for c in factors.columns if c != "RF"]
    dates = sorted(set(returns.index) & set(factors.index))

    # ── Pass 1: Rolling betas (36-month window) ────────────────────────────
    print("  Pass 1: Estimating rolling betas...")
    rolling_betas = {}  # date -> DataFrame(ticker x factors)
    window = 36

    for i, date in enumerate(dates):
        if i < window:
            continue
        window_dates = dates[i - window: i]
        r_window = returns.loc[window_dates]
        f_window = factors.loc[window_dates]
        rf_window = rf.loc[window_dates]

        betas_t = {}
        for ticker in returns.columns:
            y = r_window[ticker] - rf_window
            aligned = pd.concat([y, f_window[factor_cols]], axis=1).dropna()
            if len(aligned) < 24:
                continue
            try:
                X = sm.add_constant(aligned[factor_cols])
                model = sm.OLS(aligned.iloc[:, 0], X).fit()
                betas_t[ticker] = {f: model.params[f] for f in factor_cols
                                   if f in model.params}
            except Exception:
                continue
        rolling_betas[date] = pd.DataFrame(betas_t).T

    # ── Pass 2: Monthly cross-sectional regressions ────────────────────────
    print("  Pass 2: Running cross-sectional regressions...")
    cs_gammas = []  # one row per month: [const, beta_factor1, ...]

    for date in dates[window:]:
        if date not in rolling_betas:
            continue
        betas_t = rolling_betas[date]
        if betas_t.empty:
            continue

        # dependent variable: excess return this month
        r_t = returns.loc[date] - rf.loc[date]
        y = r_t.reindex(betas_t.index).dropna()
        X_t = betas_t.reindex(y.index).dropna()
        common = y.index.intersection(X_t.index)
        y, X_t = y.loc[common], X_t.loc[common]

        if len(y) < 30:
            continue
        try:
            X_cs = sm.add_constant(X_t)
            model = sm.OLS(y, X_cs).fit()
            row = {"date": date, "const": model.params.get("const", np.nan)}
            row.update({f: model.params.get(f, np.nan) for f in factor_cols})
            row["n_stocks"] = len(y)
            row["cs_r2"] = model.rsquared
            cs_gammas.append(row)
        except Exception:
            continue

    gammas = pd.DataFrame(cs_gammas).set_index("date")
    print(f"  Cross-sectional regressions: {len(gammas)} months")

    # ── Newey-West corrected inference ─────────────────────────────────────
    cols = ["const"] + factor_cols
    results = {}
    for col in cols:
        if col not in gammas.columns:
            continue
        g = gammas[col].dropna()
        mean_g = g.mean()
        # Newey-West SE
        model_nw = sm.OLS(g, np.ones(len(g))).fit(cov_type="HAC",
                                                    cov_kwds={"maxlags": lag})
        se_nw = float(model_nw.bse.iloc[0])
        t_stat = mean_g / se_nw
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(g) - 1))
        results[col] = {
            "mean": mean_g,
            "se_nw": se_nw,
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
        }

    summary = pd.DataFrame(results).T
    avg_cs_r2 = gammas["cs_r2"].mean()
    avg_n = gammas["n_stocks"].mean()
    print(f"  Avg cross-sectional R²: {avg_cs_r2:.3f} | Avg N: {avg_n:.0f}")

    return {
        "summary": summary,
        "gammas": gammas,
        "rolling_betas": rolling_betas,
        "avg_cs_r2": avg_cs_r2,
    }


# ── GRS Test ─────────────────────────────────────────────────────────────────

def grs_test(returns: pd.DataFrame, factors: pd.DataFrame,
             n_portfolios: int = 25) -> dict:
    """
    Gibbons-Ross-Shanken (1989) test: are all alphas jointly zero?
    Uses size-B/M 25 portfolios (constructed from our universe via sorting).
    """
    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0, index=factors.index)
    factor_cols = [c for c in factors.columns if c != "RF"]

    # Construct portfolios by sorting on size and prior-year return (as proxies)
    # Sort stocks into quintiles on two dimensions, form 25 portfolios
    dates = sorted(set(returns.index) & set(factors.index))
    common_returns = returns.reindex(dates).reindex(columns=returns.columns)

    # Use median splits to form equal-weighted portfolios
    # Dimension 1: average return (size proxy), Dimension 2: volatility (B/M proxy)
    avg_ret = common_returns.mean()
    avg_vol = common_returns.std()

    size_q = pd.qcut(avg_ret, 5, labels=[1, 2, 3, 4, 5])
    bm_q   = pd.qcut(avg_vol, 5, labels=[1, 2, 3, 4, 5])

    portfolios = {}
    for s in [1, 2, 3, 4, 5]:
        for b in [1, 2, 3, 4, 5]:
            label = f"S{s}B{b}"
            members = [t for t in returns.columns
                       if size_q.get(t) == s and bm_q.get(t) == b]
            if members:
                portfolios[label] = common_returns[members].mean(axis=1)

    port_df = pd.DataFrame(portfolios).reindex(dates)

    # Run time-series regressions for each portfolio
    T = len(dates)
    N = len(portfolios)
    K = len(factor_cols)

    alphas, resids = [], []
    Sigma_rows = []

    rf_s = rf.reindex(dates)
    X = sm.add_constant(factors[factor_cols].reindex(dates))

    for label, p_ret in port_df.items():
        y = (p_ret - rf_s).dropna()
        X_align = X.reindex(y.index).dropna()
        common_idx = y.index.intersection(X_align.index)
        if len(common_idx) < 30:
            continue
        model = sm.OLS(y.loc[common_idx], X_align.loc[common_idx]).fit()
        alphas.append(model.params["const"])
        resids.append(model.resid)

    if not alphas:
        return {"grs_stat": np.nan, "p_value": np.nan, "n_portfolios": 0}

    alpha_vec = np.array(alphas)
    resid_mat = np.column_stack(resids)
    Sigma = np.cov(resid_mat.T)

    # Factor sample moments
    f_mat = factors[factor_cols].reindex(dates).dropna().values
    mu_f = f_mat.mean(axis=0)
    Omega = np.cov(f_mat.T)

    N_p = len(alphas)
    sharpe_sq = float(mu_f @ np.linalg.inv(Omega) @ mu_f)

    try:
        Sigma_inv = np.linalg.inv(Sigma)
        grs_stat = (T / N_p) * ((T - N_p - K) / (T - K - 1)) * \
                   float(alpha_vec @ Sigma_inv @ alpha_vec) / (1 + sharpe_sq)
        p_value = 1 - stats.f.cdf(grs_stat, N_p, T - N_p - K)
    except np.linalg.LinAlgError:
        grs_stat, p_value = np.nan, np.nan

    return {
        "grs_stat": grs_stat,
        "p_value": p_value,
        "n_portfolios": N_p,
        "mean_alpha": float(alpha_vec.mean()),
        "reject_H0": p_value < 0.05 if not np.isnan(p_value) else None,
    }


# ── Model Comparison ─────────────────────────────────────────────────────────

def compare_models(returns: pd.DataFrame,
                   ff5_factors: pd.DataFrame,
                   hm_factor: pd.Series) -> dict:
    """
    Compare FF5 vs FF5+HM on:
      - Average cross-sectional R²
      - GRS test p-value (higher = alphas closer to zero = better model)
      - Mean absolute alpha
    """
    # Align
    common_idx = returns.index.intersection(ff5_factors.index).intersection(hm_factor.index)
    r  = returns.reindex(common_idx)
    f5 = ff5_factors.reindex(common_idx)
    hm = hm_factor.reindex(common_idx)

    factors_ff5   = f5
    factors_ff5hm = pd.concat([f5, hm.rename("HM")], axis=1)

    print("\n── Model 1: Fama-French 5-Factor ──────────────────────────────────")
    fm5 = fama_macbeth(r, factors_ff5)
    betas5 = estimate_ts_betas(r, factors_ff5)
    grs5   = grs_test(r, factors_ff5)

    print("\n── Model 2: FF5 + Hiring Momentum ─────────────────────────────────")
    fm6 = fama_macbeth(r, factors_ff5hm)
    betas6 = estimate_ts_betas(r, factors_ff5hm)
    grs6   = grs_test(r, factors_ff5hm)

    r2_improvement = (fm6["avg_cs_r2"] - fm5["avg_cs_r2"]) / fm5["avg_cs_r2"] * 100

    return {
        "ff5":    {"fm": fm5,  "betas": betas5, "grs": grs5},
        "ff5_hm": {"fm": fm6,  "betas": betas6, "grs": grs6},
        "r2_improvement_pct": r2_improvement,
        "hm_significant": fm6["summary"].loc["HM", "significant"]
                          if "HM" in fm6["summary"].index else False,
    }


if __name__ == "__main__":
    from data_loader import load_ff5_factors, load_stock_returns, load_hiring_momentum_factor
    import os
    os.chdir("/home/user/workspace/factor-model")

    ff5     = load_ff5_factors()
    returns = load_stock_returns()
    hm      = load_hiring_momentum_factor(ff5.index)

    comparison = compare_models(returns, ff5, hm)
    print("\n── Summary ─────────────────────────────────────────────────────────")
    print("FF5 Fama-MacBeth results:")
    print(comparison["ff5"]["fm"]["summary"].round(4))
    print("\nFF5+HM Fama-MacBeth results:")
    print(comparison["ff5_hm"]["fm"]["summary"].round(4))
    print(f"\nR² improvement from HM factor: {comparison['r2_improvement_pct']:.1f}%")
    print(f"HM factor statistically significant: {comparison['hm_significant']}")
    print(f"\nGRS test (FF5):    stat={comparison['ff5']['grs']['grs_stat']:.3f}, "
          f"p={comparison['ff5']['grs']['p_value']:.3f}")
    print(f"GRS test (FF5+HM): stat={comparison['ff5_hm']['grs']['grs_stat']:.3f}, "
          f"p={comparison['ff5_hm']['grs']['p_value']:.3f}")
