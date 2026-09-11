# Monthly engine conventions

`returns.loc[t]` is the total return from the previous valuation to valuation t. `targets.loc[t]` is traded at t after that return is earned. Signals must be known before the execution close. The engine enforces lagged positions, not the information content of arbitrary user-provided targets.

At each row:

1. Mark previous dollar holdings using total returns.
2. Accrue cash interest and short-borrow costs on starting balances.
3. Solve `fee = cost_rate * sum(abs(target * (pre_trade_nav - fee) - drifted_positions))`.
4. Set positions to target weights times after-fee NAV and cash to the residual.
5. Record equity, net/gross return, costs, turnover and exposures.

Gross traded notional includes both sales and purchases; turnover is not divided by two. The first allocation incurs fees without earning that row's return. The last row leaves holdings open unless the target is zero; reports therefore do not assume terminal liquidation. Period slices carry existing holdings through boundaries; they are not fresh strategy launches.

```python
import pandas as pd
from real_research.engine import run, Costs, metrics

dates = pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31'])
r = pd.DataFrame({'A': [0.0, 0.10, -1.0]}, index=dates)
w = pd.DataFrame({'A': [0.5, 0.5, 0.0]}, index=dates)
result = run(r, w, Costs(commission_bps=5, slippage_bps=5))
print(result.ledger)
print(metrics(result.ledger))
```

`NaN` across an entire target row means hold without rebalancing. Zero means cash/liquidation. Mixed missing and finite target weights fail. Held assets require a finite total return. Returns below -100%, nonchronological rows, misaligned axes, excess target leverage and nonpositive NAV fail.

Sharpe and Sortino use zero as the risk-free hurdle. Beta uses benchmark covariance; annualized alpha is the monthly regression intercept times 12, not a geometric excess return. For a nonzero risk-free series extend both return transformations and cash accounting consistently. Monthly borrow and cash interest use annual rate divided by 12.

The portfolio has aggregate cash, including short proceeds. It does not model separate collateral restrictions, financing spreads or intramonth margin calls. Gross target limits do not cap drift between valuations. Capacity, slippage models tied to volume, locates/recalls, taxes and event-by-event corporate actions remain outside scope.
