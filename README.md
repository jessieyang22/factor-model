# Real-Data SaaS Factor Research + Hiring Analysis

**Twelve actual companies. Yahoo adjusted prices. SEC-reported fundamentals. Indeed hiring observations. No generated data or fallback.**

This repository connects three projects: a four-signal equity study, an aggregate-hiring forecasting study, and a monthly portfolio-accounting engine. Data runs through **August 2026**. The companion [hiring-momentum repository](https://github.com/jessieyang22/hiring-momentum) contains the same version of the shared implementation and emphasizes the hiring study.

> The previous FF5 + hiring-proxy result claims are withdrawn. That implementation generated a hiring factor and treated normalized hiring scores as mock returns. Its figures and notebook results have been replaced. Prior versions remain in Git history; they are not evidence of an empirical hiring premium.

## Open the work

| Project | Research and implementation |
| --- | --- |
| Equity factors | [Research report](reports/factor_research.md) · [PDF](reports/factor_research.pdf) · [Analysis code](real_research/analyze.py) |
| Hiring and revenue forecasts | [Research report](reports/hiring_research.md) · [PDF](reports/hiring_research.pdf) · [Actual forecast scores](reports/forecast_scores.csv) |
| Backtesting framework | [Engine guide](docs/ENGINE.md) · [Engine code](real_research/engine.py) · [Tests using observed data](tests/test_real_research.py) |

![Observed equity portfolio results](reports/equity_curves.png)

## What is actually being tested

**Equities:** ADBE, CRM, NOW, HUBS, DDOG, SNOW, WDAY, VEEV, CRWD, NET, MDB and ZS. Four fixed signals: 12-to-1-month price momentum, low volatility, SEC-reported operating margin and year-over-year revenue growth. Cross-sectional winsorization and standardization, equal composite weights, monthly tercile portfolios. No value signal is claimed because historical valuation inputs were not constructed.

**Hiring:** Indeed's **U.S. software-development job-postings index**, distributed through FRED. It is an occupation-wide series, not firm-level job postings. The test asks whether that hiring backdrop improves SaaS revenue forecasts beyond reported growth, margin, price momentum and company indicators. A separate SPY/cash rule tests aggregate hiring-based market timing.

**Framework:** Actual adjusted-price returns drive a self-financing cash-and-holdings ledger, monthly weight drift, transaction costs, borrowing costs and exposure/performance diagnostics. The same engine evaluates all portfolios.

## Results, without hiding the weak ones

The chronological test is January 2023-August 2026. These are retrospective historical calculations after modeled costs, not a live track record.

| Portfolio | Annualized return | Sharpe, zero hurdle | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Composite long/short | -6.38% | -0.34 | -31.21% |
| Composite long-only | 23.07% | 0.74 | -40.17% |
| Eligible SaaS equal-weight | 29.92% | 0.89 | -38.47% |
| SPY buy-and-hold | 22.39% | 1.68 | -8.33% |

The long-only composite **underperforms the eligible-universe equal-weight comparison**. This run does not support claiming a composite stock-selection advantage.

Across 165 scored company/quarter forecasts in the test window, baseline RMSE is 0.1057 versus 0.0851 with aggregate hiring features (revenue growth in decimal units). This is descriptive improvement in a small, revised-vintage sample. The common hiring input means 165 company rows are not 165 independent macro observations.

## Run it

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m real_research.analyze
```

Committed, normalized **real observations** allow an offline run after installing dependencies. To refresh the underlying public sources:

```bash
python -m real_research.download --refresh
python -m real_research.analyze
```

The downloader raises errors if a source fails. It never switches to generated data. Refreshing current-vintage sources may change results; the fixed August 2026 cutoff does not freeze provider revisions.

## Sources and audit

- [Yahoo Finance](https://finance.yahoo.com/): adjusted monthly prices from the chart endpoint, starting January 2015 or IPO. Endpoints and original response hashes are in [manifest.json](data/real/manifest.json).
- [SEC EDGAR Company Facts](https://www.sec.gov/search-filings/edgar-application-programming-interfaces): reported quarterly revenue and operating income. [Fundamental rows](data/real/fundamentals.csv) retain filing accessions, revenue tags and Q4 derivation flags.
- [Indeed via FRED: IHLIDXUSTPSOFTDEVE](https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE): software-development job postings. Attribution: Indeed, retrieved via FRED, Federal Reserve Bank of St. Louis. [Indeed methodology and CC BY 4.0 license](https://github.com/hiring-lab/job_postings_tracker).

[Audit output](reports/audit.json) records input hashes, source retrieval timestamp, coverage and execution assumptions. No API keys are required. Full raw responses stay in the ignored `data/real/raw/` directory; normalized observations are committed for reproducibility.

## Limits that change the interpretation

1. This is a selected **surviving-company SaaS sample**, not historical S&P 500 membership. It excludes failed/acquired firms and has concentrated technology exposure.
2. Indeed revises its historical series. A one-month feature lag does **not** reconstruct historical availability. Treat the hiring analysis as retrospective, not vintage-correct out-of-sample evidence.
3. SEC quarters retain first reports. Q4 may be annual minus previously filed nine-month values, which can mix accounting vintages. Those derivations and accessions are visible for review.
4. Yahoo's current adjusted prices approximate total returns; there is no complete delisting panel, corporate-action ledger, stock-loan inventory or volume-based market-impact model.
5. Costs are 10 bps per dollar traded plus 100 bps annual short borrowing. Cash earns zero. Test splits are chronological but retrospectively chosen; no prospective alpha claim or multiple-testing-adjusted significance claim is made.

See [methodology](docs/METHODOLOGY.md), [validation](docs/VALIDATION.md), and the reports before using the numbers in a pitch or resume. Built with AI assistance; the source and limitations are exposed for independent review.
