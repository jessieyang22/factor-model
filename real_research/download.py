"""Fetch Yahoo adjusted prices, SEC reported fundamentals and Indeed/FRED hiring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np
import pandas as pd

UNIVERSE={'ADBE':796343,'CRM':1108524,'NOW':1373715,'HUBS':1404655,
          'DDOG':1561550,'SNOW':1640147,'WDAY':1327811,'VEEV':1393052,
          'CRWD':1535527,'NET':1477333,'MDB':1441816,'ZS':1713683}
REVENUE=['RevenueFromContractWithCustomerExcludingAssessedTax','RevenueFromContractWithCustomerIncludingAssessedTax','Revenues','SalesRevenueNet','SalesRevenueServicesNet']
SERIES='IHLIDXUSTPSOFTDEVE'


def retrieve(url,path,refresh=False):
    if path.exists() and not refresh:return path.read_bytes()
    request=urllib.request.Request(url,headers={'User-Agent':'JessieYangResearch/0.2 (github.com/jessieyang22/factor-model)'})
    with urllib.request.urlopen(request,timeout=45) as response:body=response.read()
    if not body:raise ValueError(f'Empty source: {url}')
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(body);tmp.replace(path)
    return body


def observations(facts,tags):
    """Preserve filing dates and accession numbers; normalize no fabricated values."""
    rows=[]
    for priority,tag in enumerate(tags):
        for r in facts.get(tag,{}).get('units',{}).get('USD',[]):
            if {'start','end','filed','val','accn'}<=r.keys() and r.get('form') in {'10-Q','10-K','10-Q/A','10-K/A'}:
                rows.append({**{k:r[k] for k in ['start','end','filed','val','accn']},'tag':tag,'priority':priority})
    if not rows:return pd.DataFrame()
    d=pd.DataFrame(rows)
    for c in ['start','end','filed']:d[c]=pd.to_datetime(d[c])
    d['days']=(d.end-d.start).dt.days+1
    return d.sort_values(['filed','priority']).drop_duplicates(['start','end','filed'])


def quarterly(d):
    """Direct quarters plus Q4=annual minus earlier published 9-month cumulative.

    Uses first reports, never revised future versions. Q4 can mix reporting
    vintages when nine-month comparatives are absent in the annual filing;
    derivation and both source accessions remain visible for review.
    """
    if d.empty:return d
    direct=d[d.days.between(70,110)].copy();direct['method']='direct_quarter';direct['subtract_accn']=''
    out=[direct]
    for _,annual in d[d.days.between(330,400)].iterrows():
        nine=d[(d.start==annual.start)&d.days.between(240,300)&(d.filed<=annual.filed)&(d.end<annual.end)]
        if nine.empty:continue
        nine=nine.sort_values('filed').iloc[-1]
        row=annual.copy();row['start']=nine.end+pd.Timedelta(days=1)
        row['val']=annual.val-nine.val;row['days']=(row.end-row.start).days+1
        row['method']='annual_minus_9m';row['subtract_accn']=nine.accn
        if 70<=row['days']<=110:out.append(pd.DataFrame([row]))
    result=pd.concat(out,ignore_index=True)
    return result.sort_values(['filed','priority','method']).drop_duplicates('end',keep='first').sort_values('end')


def company_quarters(raw,ticker):
    f=json.loads(raw)['facts'].get('us-gaap',{})
    rev=quarterly(observations(f,REVENUE));op=quarterly(observations(f,['OperatingIncomeLoss']))
    if rev.empty:return pd.DataFrame()
    rows=[]
    for _,r in rev.iterrows():
        prior=rev[((r.end-rev.end).dt.days.between(330,400))&(rev.filed<=r.filed)]
        if prior.empty:continue
        previous=prior.iloc[-1]
        if previous.val<=0 or r.val<=0:continue
        operating=op[(op.end==r.end)&(op.start==r.start)&(op.filed<=r.filed)] if not op.empty else op
        margin=operating.iloc[-1].val/r.val if len(operating) else np.nan
        rows.append(dict(ticker=ticker,period_end=r.end,available_at=r.filed+pd.Timedelta(days=1),
            revenue=r.val,revenue_growth=r.val/previous.val-1,operating_margin=margin,
            accession=r.accn,prior_accession=previous.accn,revenue_tag=r.tag,
            derivation=r.method,subtract_accession=r.subtract_accn,
            operating_accession=operating.iloc[-1].accn if len(operating) else ''))
    return pd.DataFrame(rows)


def download(root='data/real',refresh=False):
    root=Path(root);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    fetched=datetime.now(timezone.utc).isoformat();records=[];prices=[];quarters=[]
    start=int(datetime(2015,1,1,tzinfo=timezone.utc).timestamp());stop=int(datetime(2026,9,1,tzinfo=timezone.utc).timestamp())
    for ticker,cik in UNIVERSE.items():
        url=f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
        blob=retrieve(url,raw/f'{ticker}_sec.json',refresh)
        q=company_quarters(blob,ticker);quarters.append(q)
        records.append(dict(name=f'{ticker}_sec',url=url,sha256=hashlib.sha256(blob).hexdigest(),rows=len(q)))
        print(ticker,'reported growth quarters',len(q),flush=True)
        time.sleep(.15)
    def price(ticker):
        url=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={stop}&interval=1mo'
        blob=retrieve(url,raw/f'{ticker}_prices.json',refresh)
        result=json.loads(blob)['chart']['result'][0]
        dates=pd.to_datetime(result['timestamp'],unit='s',utc=True).tz_localize(None).to_period('M').to_timestamp('M')
        p=pd.DataFrame({'date':dates,'ticker':ticker,'adjusted_close':result['indicators']['adjclose'][0]['adjclose']})
        p=p.dropna();p=p[(p.date>='2015-01-01')&(p.date<='2026-08-31')]
        if p.empty or (p.adjusted_close<=0).any():raise ValueError(f'Invalid actual prices: {ticker}')
        return p,dict(name=f'{ticker}_prices',url=url,sha256=hashlib.sha256(blob).hexdigest(),rows=len(p))
    with ThreadPoolExecutor(max_workers=3) as executor:
        for p,record in executor.map(price,[*UNIVERSE,'SPY']):prices.append(p);records.append(record)
    url=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES}'
    blob=retrieve(url,raw/'indeed_software.csv',refresh)
    h=pd.read_csv(raw/'indeed_software.csv',parse_dates=['observation_date']).rename(columns={'observation_date':'date',SERIES:'hiring_index'})
    h['hiring_index']=pd.to_numeric(h.hiring_index,errors='coerce');h=h.dropna()
    records.append(dict(name='Indeed software development postings via FRED',url=url,sha256=hashlib.sha256(blob).hexdigest(),rows=len(h)))
    pd.concat(prices).sort_values(['date','ticker']).to_csv(root/'prices.csv',index=False)
    q=pd.concat(quarters).sort_values(['period_end','ticker'])
    q[q.period_end<='2026-08-31'].to_csv(root/'fundamentals.csv',index=False)
    h.to_csv(root/'hiring.csv',index=False)
    manifest=dict(data_mode='historical',retrieved_at=fetched,universe=UNIVERSE,sources=records,
        analysis_end='2026-08-31',price_start='2015-01-01',hiring_series=SERIES,
        limitations=['Purposively selected surviving SaaS companies, not historical S&P 500 membership',
            'Yahoo adjusted prices are latest-vintage, approximate dividend-adjusted total returns',
            'Indeed software hiring is an occupation-wide index, not company-specific hiring',
            'Indeed history is revised; 1-month lag does not reconstruct historical data vintages',
            'SEC first-report quarters retained; Q4 annual-minus-9m derivations may mix accounting vintages',
            'No analyst estimates, company job-posting history, stock-loan availability or delisting panel'])
    manifest['normalized_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csv')}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Real data saved:',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/real');p.add_argument('--refresh',action='store_true')
    a=p.parse_args();download(a.data,a.refresh)
