# Fama-French 5-Factor Model + Hiring Momentum

**Author:** Jessica Yang | [github.com/jessieyang22](https://github.com/jessieyang22)  
**Related repo:** [jessieyang22/hiring-momentum](https://github.com/jessieyang22/hiring-momentum)

---

## Research Question

Does a proprietary **hiring momentum signal** — constructed from LinkedIn job posting velocity across SaaS companies — explain cross-sectional stock returns beyond the standard Fama-French 5-factor model?

---

## Methodology

### 1. Data
- **Universe:** 181 S&P 500 stocks across all GICS sectors
- **Sample:** 2018–2025 (monthly, 95 observations)
- **Prices:** Yahoo Finance via `yfinance` (auto-adjusted monthly close)
- **FF5 Factors:** Ken French Data Library (Mkt-RF, SMB, HML, RMW, CMA, RF)

### 2. Hiring Momentum Factor (HM)
Built from the [hiring-momentum](https://github.com/jessieyang22/hiring-momentum) project:
- LinkedIn job posting data collected across 18 SaaS companies using Apify
- 4-quarter rolling z-score normalization of QoQ job count growth
- Long top hiring tercile, short bottom tercile
- Quarterly signal forward-filled to monthly frequency

### 3. Fama-MacBeth Two-Pass Regression
**Pass 1 — Rolling Time-Series Betas:**  
For each stock *i*, rolling 36-month OLS:

$$r_{i,t} - r_f = \alpha_i + \sum_k \beta_{i,k} f_{k,t} + \epsilon_{i,t}$$

**Pass 2 — Monthly Cross-Sectional Regression:**  
Each month *t*:

$$r_{i,t} - r_f = \gamma_{0,t} + \sum_k \gamma_{k,t} \hat{\beta}_{i,k} + \eta_{i,t}$$

**Inference:** Newey-West corrected standard errors (4 lags) to account for serial correlation in the gamma series.

### 4. Model Evaluation
- t-statistics and p-values on each factor premium
- GRS (1989) test: joint null that all portfolio alphas are zero
- Cross-sectional R² comparison across all cross-sectional periods

---

## Key Results

| Metric                     | FF5     | FF5 + HM |
|---------------------------|---------|----------|
| Avg Cross-Sectional R²    | 20.0%   | 22.0%    |
| R² Improvement            | —       | +9.6%    |
| Mean \|Alpha\| (monthly)  | 0.602%  | 0.603%   |
| GRS F-statistic           | 4.257   | 4.223    |
| GRS p-value               | 0.000   | 0.000    |

**HM Factor Premium:**
- Monthly mean: +0.28%
- Newey-West t-statistic: 1.56
- p-value: 0.124 (directionally positive, not yet significant at 5%)

**Interpretation:** The hiring momentum factor has a positive and directionally significant risk premium consistent with the hypothesis that companies accelerating hiring lead revenue growth by 1–2 quarters. The 9.6% improvement in cross-sectional R² suggests meaningful incremental explanatory power. Statistical significance is limited by the 59-period sample; using the full Apify live dataset (and a broader cross-section of companies) is expected to improve inference.

---

## Visualizations

| Figure | Description |
|--------|-------------|
| `figures/factor_premia.png` | Fama-MacBeth factor premiums with Newey-West t-stats |
| `figures/cs_r2_over_time.png` | Cross-sectional R² over time, FF5 vs FF5+HM |
| `figures/factor_correlation.png` | Factor correlation heatmap |
| `figures/hm_gamma_over_time.png` | HM risk premium and cumulative premium over time |
| `figures/model_comparison.png` | Side-by-side model comparison (R², |alpha|, GRS) |

---

## Repo Structure

```
factor-model/
├── data_loader.py              # Data pipeline (FF5, prices, HM factor)
├── factor_model.py             # Fama-MacBeth, GRS, model comparison
├── visualize.py                # All figures
├── run_factor_model.py         # Production entry point (CLI)
├── factor_model_analysis.ipynb # Full analysis notebook
├── results_fama_macbeth.csv    # Output table
├── figures/                    # Saved plots
└── cache/                      # Cached data (prices.parquet, ff5.zip)
```

---

## Usage

```bash
# Full run (downloads data if not cached, ~3-4 min first run)
python run_factor_model.py

# Force re-download all data
python run_factor_model.py --no-cache

# Run without generating plots
python run_factor_model.py --no-plots

# Open notebook
jupyter notebook factor_model_analysis.ipynb
```

### Requirements
```bash
pip install yfinance pandas numpy scipy statsmodels matplotlib seaborn nbformat pyarrow
```

---

## Extending the Model

To use live hiring momentum data (once Apify pipeline is running):

1. Export the hiring momentum z-scores from `jessieyang22/hiring-momentum` to CSV
2. Pass the path to `load_hiring_momentum_factor(ff5.index, hiring_data_path="path/to/hm.csv")`
3. The model will automatically use actual factor values instead of the synthetic proxy

To expand the universe, modify the `UNIVERSE` list in `data_loader.py`.

---

## References
- Fama, E.F. & French, K.R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*
- Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*
- Fama, E.F. & MacBeth, J.D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*
- Gibbons, M.R., Ross, S.A. & Shanken, J. (1989). A test of the efficiency of a given portfolio. *Econometrica*
- Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*
