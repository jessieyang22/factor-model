# Data and evaluation conventions

## Actual observations

`prices.csv`: date, ticker, adjusted_close. One month-end label per actual monthly source price. The selected end is August 31, 2026. IPO histories are shorter; missing history is excluded rather than generated.

`fundamentals.csv`: reported quarterly revenue, year-over-year growth, operating margin, period end, availability date, accession identifiers and quarter derivation. Availability is SEC filing date plus one day. Features require availability strictly before the decision date. Only the first reported quarter is retained.

`hiring.csv`: actual daily software-development postings index observations, sourced from Indeed via FRED. Monthly last observations become a common time-series predictor. No company identifiers are attached to this index because it does not measure a particular company's openings.

The manifest hashes every normalized CSV. Loaders reject modified inputs until a proper refresh regenerates the provenance record. Source URLs and raw response hashes are recorded; raw source bodies are cached locally, not included as multi-megabyte filing dumps.

## Quarter reconciliation

Revenue tags are tried in a stated order. Direct quarter duration is 70-110 days. Annual duration is 330-400 days. A derived Q4 subtracts a 240-300-day YTD value with the same fiscal start, published no later than the annual value. Both accessions are retained. This arithmetic may mix reporting vintages; review it when auditing real conclusions. Growth uses a prior-year quarter whose end is 330-400 days earlier and whose publication was already known. Missing margin observations remain missing.

The revised analysis uses operating margin for profitability and reported revenue growth. It does not claim ROE, FCF yield, historical market capitalization, earnings surprises, estimate revisions or company hiring counts that it has not sourced.

## Portfolio construction

Each date separately: winsorize at 5/95%, z-score, average four complete signals, sort terciles. At least six observed names are required. Long-short gross exposure is 1, with +0.5/-0.5 books; long-only gross is 1. Equal-weight comparator uses the same complete-factor universe. SPY is also evaluated. Small sample + technology concentration prevent a sector-neutral broad-equity interpretation.

Price signals lag one complete month. Reports distinguish development through 2022 from chronological test starting 2023. No weights are optimized from test returns. The universe and split were chosen retrospectively, and source histories may be revised: chronological separation alone does not make this a prospective validation.

## Forecasts

Fixed ridge penalty 1; train-only scaling; company indicator effects; baseline financial/price controls versus the same controls plus hiring growth and acceleration. Only reported labels are eligible for training. Latest eligible feature row per company/target quarter avoids repeated training copies. Evaluation similarly retains one last prediction per company/target quarter/model, and only targets reported by the cutoff.

All companies share the aggregate hiring predictor, limiting effective sample size. Return regressions use a dynamic equal-weight available-company basket less SPY and HAC errors at 1/3/6-month horizons. These are exploratory descriptive tests, not a company hiring factor or a causal conclusion.
