"""Monthly, self-financing portfolio accounting with explicit end-of-period trades."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    annual_borrow_bps: float = 100.0
    annual_cash_rate: float = 0.0
    max_gross: float = 2.0

    def __post_init__(self):
        if not all(np.isfinite(x) for x in vars(self).values()):
            raise ValueError("Costs must be finite")
        if min(self.commission_bps, self.slippage_bps, self.annual_borrow_bps) < 0:
            raise ValueError("Costs cannot be negative")
        if self.max_gross <= 0 or (self.commission_bps + self.slippage_bps) / 10000 * self.max_gross >= 1:
            raise ValueError("Invalid gross leverage / cost combination")


@dataclass
class Backtest:
    ledger: pd.DataFrame
    weights: pd.DataFrame


def run(returns: pd.DataFrame, targets: pd.DataFrame, costs: Costs = Costs(),
        initial_nav: float = 1.0) -> Backtest:
    """Earn interval t returns on holdings from t-1; trade targets[t] at t close.

    targets[t] must contain information known BEFORE that close. NaN for an
    entire target row means hold (no rebalance); a zero row means liquidate.
    Returns must include distributions and delisting proceeds. No implicit
    forward filling, missing-return liquidation, or survivorship repair.
    """
    if initial_nav <= 0 or not np.isfinite(initial_nav):
        raise ValueError("initial_nav must be positive")
    if returns.empty or returns.index.has_duplicates or returns.columns.has_duplicates:
        raise ValueError("Nonempty unique return axes required")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must be chronological")
    if not targets.index.equals(returns.index) or not targets.columns.equals(returns.columns):
        raise ValueError("Targets must align exactly with returns")
    if np.isinf(returns.to_numpy()).any() or (returns < -1).any().any():
        raise ValueError("Invalid total return")
    pos = np.zeros(returns.shape[1]); cash = float(initial_nav)
    ledger, weights = [], []
    rate = (costs.commission_bps + costs.slippage_bps) / 10000
    for date, row in returns.iterrows():
        start = cash + pos.sum()
        r = row.to_numpy(float)
        if ((np.abs(pos) > 1e-12) & np.isnan(r)).any():
            raise ValueError(f"Missing return for held asset at {date}")
        borrow = np.maximum(-pos, 0).sum() * costs.annual_borrow_bps / 10000 / 12
        interest = cash * costs.annual_cash_rate / 12
        pnl = np.dot(pos, np.nan_to_num(r)) + interest
        pos *= 1 + np.nan_to_num(r)
        cash += interest - borrow
        before = cash + pos.sum()
        if before <= 0:
            raise ValueError(f"Portfolio insolvent at {date}")
        target = targets.loc[date].to_numpy(float)
        fee = turnover = 0.0
        if not np.isnan(target).all():
            if not np.isfinite(target).all():
                raise ValueError("Target rows must be finite or entirely NaN")
            if np.abs(target).sum() > costs.max_gross + 1e-10:
                raise ValueError("Gross leverage exceeds limit")
            # Solve fee = cost_rate * traded notional, using AFTER-fee target NAV.
            lo, hi = 0.0, before
            for _ in range(64):
                fee = (lo + hi) / 2
                need = rate * np.abs(target * (before - fee) - pos).sum()
                if fee < need: lo = fee
                else: hi = fee
            after = before - fee
            if after <= 1e-12:
                raise ValueError("Transaction costs consume portfolio")
            new_pos = target * after
            turnover = np.abs(new_pos - pos).sum() / before
            pos = new_pos
            cash = after - pos.sum()
        nav = cash + pos.sum()
        ledger.append(dict(date=date, nav=nav, net_return=nav/start-1,
                           gross_return=pnl/start, trading_cost=fee/start,
                           borrow_cost=borrow/start, turnover=turnover,
                           cash_weight=cash/nav, gross_exposure=np.abs(pos).sum()/nav,
                           net_exposure=pos.sum()/nav))
        weights.append(pos/nav)
    return Backtest(pd.DataFrame(ledger).set_index("date"),
                    pd.DataFrame(weights, index=returns.index, columns=returns.columns))


def metrics(ledger: pd.DataFrame, benchmark: pd.Series | None = None) -> dict:
    r = ledger.net_return.astype(float)
    if len(r) == 0: return {}
    wealth = (1 + r).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    sd = r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r, 0)**2))
    out = dict(periods=len(r), cumulative_return=wealth.iloc[-1]-1,
               annualized_return=wealth.iloc[-1]**(12/len(r))-1,
               annualized_volatility=sd*np.sqrt(12),
               sharpe=r.mean()/sd*np.sqrt(12) if sd > 0 else np.nan,
               sortino=r.mean()/downside*np.sqrt(12) if downside > 0 else np.nan,
               max_drawdown=(wealth/peak-1).min(), hit_rate=(r>0).mean(),
               annualized_turnover=ledger.turnover.mean()*12,
               trading_cost_sum=ledger.trading_cost.sum(), borrow_cost_sum=ledger.borrow_cost.sum())
    if benchmark is not None:
        both = pd.concat([r.rename('r'), benchmark.rename('b')], axis=1).dropna()
        if len(both)>2 and both.b.var()>0:
            beta = both.r.cov(both.b)/both.b.var()
            active = both.r-both.b
            out.update(beta=beta, annualized_alpha=(both.r.mean()-beta*both.b.mean())*12,
                       information_ratio=active.mean()/active.std()*np.sqrt(12) if active.std()>0 else np.nan)
    return {k: float(v) for k,v in out.items()}
