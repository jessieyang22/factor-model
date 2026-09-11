"""Observed-data factor portfolios and aggregate hiring research."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .engine import Costs,run,metrics

SIGNALS=['momentum','low_volatility','profitability','growth']


def load(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('data_mode')!='historical':raise ValueError('Real historical observations are required')
    for name,digest in manifest['normalized_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Input changed: {name}')
    p=pd.read_csv(root/'prices.csv',parse_dates=['date'])
    f=pd.read_csv(root/'fundamentals.csv',parse_dates=['period_end','available_at'])
    h=pd.read_csv(root/'hiring.csv',parse_dates=['date'])
    if p.duplicated(['date','ticker']).any() or f.duplicated(['period_end','ticker']).any():raise ValueError('Duplicate observed keys')
    if (p.adjusted_close<=0).any() or p.adjusted_close.isna().any():raise ValueError('Invalid prices')
    if (f.available_at<=f.period_end).any():raise ValueError('Filing precedes period end')
    return p,f,h,manifest


def zscore(x):
    x=x.replace([np.inf,-np.inf],np.nan);x=x.clip(x.quantile(.05),x.quantile(.95));sd=x.std(ddof=0)
    return (x-x.mean())/sd if sd>1e-10 else x*0


def features(prices,fundamentals,hiring):
    levels=prices.pivot(index='date',columns='ticker',values='adjusted_close').sort_index()
    returns=levels.pct_change(fill_method=None)
    momentum=levels.shift(1)/levels.shift(12)-1
    vol=returns.rolling(12,min_periods=12).std().shift(1)
    monthly=hiring.set_index('date').hiring_index.resample(pd.offsets.MonthEnd()).last()
    # Explicit observation lag; this does NOT remove revised-vintage bias.
    hiring_growth=monthly.shift(1)/monthly.shift(4)-1
    hiring_acceleration=hiring_growth-hiring_growth.shift(3)
    rows=[]
    for date in levels.index:
        known=fundamentals[(fundamentals.available_at<date)&((date-fundamentals.period_end).dt.days<550)]
        known=known.sort_values('period_end').drop_duplicates('ticker').set_index('ticker')
        for ticker in levels.columns.drop('SPY'):
            if ticker not in known.index or pd.isna(momentum.loc[date,ticker]) or pd.isna(vol.loc[date,ticker]):continue
            f=known.loc[ticker]
            rows.append(dict(date=date,ticker=ticker,momentum=momentum.loc[date,ticker],low_volatility=-vol.loc[date,ticker],
                profitability=f.operating_margin,growth=f.revenue_growth,fundamental_available_at=f.available_at,
                fundamental_period=f.period_end,hiring_growth=hiring_growth.get(date,np.nan),hiring_acceleration=hiring_acceleration.get(date,np.nan)))
    panel=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    for signal in SIGNALS:panel[signal+'_z']=panel.groupby('date')[signal].transform(zscore)
    zs=[x+'_z' for x in SIGNALS];panel['composite']=panel[zs].mean(axis=1).where(panel[zs].notna().all(axis=1))
    return panel,returns,monthly


def targets(panel,returns,signal,long_only=False):
    w=pd.DataFrame(0.,index=returns.index,columns=returns.columns)
    for date,g in panel.dropna(subset=[signal]).groupby('date'):
        g=g.sort_values([signal,'ticker'])
        if len(g)<6 or g[signal].nunique()<2:continue
        n=max(1,len(g)//3)
        if g[signal].iloc[n-1]>=g[signal].iloc[-n]:continue
        w.loc[date,g.ticker.iloc[-n:]]=(1 if long_only else .5)/n
        if not long_only:w.loc[date,g.ticker.iloc[:n]]=-.5/n
    return w


def ridge(train,test,columns):
    a=train[columns].copy();b=test[columns].copy();means=a.mean();sd=a.std(ddof=0).replace(0,1)
    a=(a-means)/sd;b=(b-means)/sd
    for ticker in sorted(train.ticker.unique())[1:]:
        a[ticker]=(train.ticker==ticker).astype(float);b[ticker]=(test.ticker==ticker).astype(float)
    x=np.column_stack([np.ones(len(a)),a]);xt=np.column_stack([np.ones(len(b)),b]);penalty=np.eye(x.shape[1]);penalty[0,0]=0
    beta=np.linalg.pinv(x.T@x+penalty)@x.T@train.target.to_numpy()
    return xt@beta


def forecasts(panel,fundamentals,end):
    parts=[]
    for ticker,g in panel.groupby('ticker'):
        f=fundamentals[fundamentals.ticker==ticker].sort_values('period_end')
        matched=pd.merge_asof(g.sort_values('date'),f[['period_end','available_at','revenue_growth']].rename(columns={
            'period_end':'target_period','available_at':'label_available_at','revenue_growth':'target'}),
            left_on='date',right_on='target_period',direction='forward',allow_exact_matches=False)
        matched=matched[(matched.target_period-matched.date).dt.days<=110]
        parts.append(matched)
    d=pd.concat(parts).dropna(subset=['growth','profitability','momentum','hiring_growth','hiring_acceleration'])
    rows=[]
    for date,g in d[d.date>='2022-01-01'].groupby('date'):
        train=d[(d.date<date)&(d.label_available_at<date)].dropna(subset=['target'])
        train=train.sort_values('date').drop_duplicates(['ticker','target_period'],keep='last')
        if len(train)<50 or train.target_period.nunique()<8:continue
        for model,cols in [('baseline',['growth','profitability','momentum']),('hiring_augmented',['growth','profitability','momentum','hiring_growth','hiring_acceleration'])]:
            for (_,row),prediction in zip(g.iterrows(),ridge(train,g,cols)):
                rows.append(dict(date=date,ticker=row.ticker,model=model,prediction=prediction,target=row.target,
                    target_period=row.target_period,label_available_at=row.label_available_at,train_n=len(train),
                    max_training_label=train.label_available_at.max()))
    predictions=pd.DataFrame(rows);scores=[]
    for split,first,last in [('development','2022-01-01','2022-12-31'),('test','2023-01-01',end)]:
        d=predictions[(predictions.date>=first)&(predictions.date<=last)&(predictions.label_available_at<=pd.Timestamp(end))].dropna(subset=['target'])
        d=d.sort_values('date').drop_duplicates(['ticker','target_period','model'],keep='last')
        for model,g in d.groupby('model'):
            error=g.prediction-g.target
            scores.append(dict(split=split,model=model,n=len(g),rmse=np.sqrt(np.mean(error**2)),mae=np.mean(abs(error))))
    return predictions,pd.DataFrame(scores)


def hac_regression(frame,target,columns,lags):
    d=frame.dropna(subset=[target,*columns]);x=np.column_stack([np.ones(len(d)),d[columns]]);y=d[target].to_numpy()
    if len(d)<len(columns)+10:return pd.DataFrame()
    inv=np.linalg.pinv(x.T@x);beta=inv@x.T@y;u=y-x@beta;s=x*u[:,None];meat=s.T@s
    for lag in range(1,min(lags,len(d)-1)+1):
        gamma=s[lag:].T@s[:-lag];meat+=(1-lag/(lags+1))*(gamma+gamma.T)
    se=np.sqrt(np.maximum(np.diag(inv@meat@inv),0))
    return pd.DataFrame({'term':['intercept',*columns],'coefficient':beta,'hac_se':se,'t':beta/se,'n':len(d),'lags':lags})


def analyze(root='data/real',output='reports'):
    from .report import report
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    prices,fund,hiring,manifest=load(root);panel,returns,monthly=features(prices,fund,hiring)
    usable=panel.dropna(subset=['composite']).groupby('date').ticker.nunique()
    first=usable[usable>=6].index.min()
    if pd.isna(first):raise ValueError('Not enough actual companies for a portfolio')
    returns=returns.loc[first:];panel=panel[panel.date>=first];end=manifest['analysis_end']
    benchmark=returns.SPY
    strategies={s:targets(panel,returns,s) for s in ['composite',*[s+'_z' for s in SIGNALS]]}
    strategies['composite_long_only']=targets(panel,returns,'composite',True)
    # Available-universe benchmark has the same history/coverage requirement.
    equal=pd.DataFrame(0.,index=returns.index,columns=returns.columns)
    for date,g in panel.dropna(subset=['composite']).groupby('date'):
        if len(g)>=6:equal.loc[date,g.ticker]=1/len(g)
    strategies['eligible_equal_weight']=equal
    w=pd.DataFrame(0.,index=returns.index,columns=returns.columns)
    bg=monthly.shift(1)/monthly.shift(4)-1
    w['SPY']=(bg.reindex(w.index)>0).astype(float);strategies['hiring_market_timing']=w
    market=pd.DataFrame(0.,index=returns.index,columns=returns.columns);market['SPY']=1.;strategies['spy_buy_hold']=market
    ledgers={};weights={};performance=[]
    for name,w in strategies.items():
        bt=run(returns,w);ledgers[name]=bt.ledger;weights[name]=bt.weights
        bt.ledger.to_csv(output/f'{name}_ledger.csv');bt.weights.to_csv(output/f'{name}_weights.csv')
        for split,a,b in [('development',str(first.date()),'2022-12-31'),('test','2023-01-01',end)]:
            selected=bt.ledger.loc[a:b]
            if len(selected):performance.append(dict(strategy=name,split=split,**metrics(selected,benchmark)))
    performance=pd.DataFrame(performance);performance.to_csv(output/'performance.csv',index=False)
    nextret=returns.shift(-1).stack().rename('next_return').reset_index();nextret.columns=['date','ticker','next_return']
    joined=panel.merge(nextret,on=['date','ticker']);ic=[]
    for date,g in joined.groupby('date'):
        for signal in ['composite',*[x+'_z' for x in SIGNALS]]:
            x=g[[signal,'next_return']].dropna()
            if len(x)>=6 and x[signal].nunique()>1:ic.append(dict(date=date,signal=signal,rank_ic=x[signal].rank().corr(x.next_return.rank()),n=len(x)))
    pd.DataFrame(ic).to_csv(output/'information_coefficients.csv',index=False)
    panel[[x+'_z' for x in SIGNALS]].corr().to_csv(output/'factor_correlations.csv')
    predictions,scores=forecasts(panel,fund,end)
    predictions.to_csv(output/'revenue_predictions.csv',index=False);scores.to_csv(output/'forecast_scores.csv',index=False)
    panel.to_csv(output/'features.csv',index=False)
    timing=pd.DataFrame({'hiring_growth':bg,'market_momentum':prices[prices.ticker=='SPY'].set_index('date').adjusted_close.pct_change(12,fill_method=None).shift(1)})
    basket=returns.drop(columns='SPY').mean(axis=1)
    for horizon in [1,3,6]:
        forward=pd.Series(1.,index=returns.index);marketf=forward.copy()
        for lag in range(1,horizon+1):forward*=1+basket.shift(-lag);marketf*=1+benchmark.shift(-lag)
        timing[f'future_{horizon}m_excess']=forward-marketf
        reg=hac_regression(timing.loc['2021':],f'future_{horizon}m_excess',['hiring_growth','market_momentum'],max(3,horizon))
        reg.to_csv(output/f'hiring_return_regression_{horizon}m.csv',index=False)
    timing.to_csv(output/'hiring_return_observations.csv')
    costs=[]
    for bps in [0,5,10,25,50]:
        bt=run(returns,strategies['composite'],Costs(commission_bps=bps,slippage_bps=0))
        costs.append(dict(one_way_bps=bps,**metrics(bt.ledger.loc['2023':],benchmark)))
    pd.DataFrame(costs).to_csv(output/'cost_sensitivity.csv',index=False)
    audit=dict(data_mode='historical',analysis_end=end,first_portfolio_date=str(first.date()),source_manifest=manifest,
        factor_rows=len(panel),real_tickers=sorted(panel.ticker.unique()),quarterly_growth_observations=len(fund),
        hiring_observations=len(hiring),prediction_rows=len(predictions),
        notes=['No generated observations, generated proxy factors or generated fallback',
               'All real-data outcomes retained regardless of direction',
               'Chronological test is not a vintage-correct out-of-sample claim',
               'Four fixed signals; no value factor because historical valuation inputs were not constructed',
               'Ridge penalty fixed at 1 before evaluating outcomes; no tuning on test'],
        assumptions=dict(one_way_bps=10,borrow_bps_annual=100,zero_cash_rate=True,gross_long_short=1.,price_lag_months=1,
                         test_start='2023-01-01',source_vintage=manifest['retrieved_at']))
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    load(root) # Detect changed inputs, never silently certify different bytes.
    report(output,audit,performance,scores,ledgers,monthly,panel)
    print(performance[performance.split=='test'][['strategy','annualized_return','sharpe','max_drawdown']].to_string(index=False))
    print(scores.to_string(index=False))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/real');p.add_argument('--output',default='reports')
    a=p.parse_args();analyze(a.data,a.output)
