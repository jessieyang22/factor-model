"""Rebuildable research PDFs and plots from computed outputs."""
from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab

FONTS = Path(reportlab.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("ResearchSans", str(FONTS / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("ResearchSansBold", str(FONTS / "VeraBd.ttf")))


def table(frame):
    style=getSampleStyleSheet()['BodyText']
    style.fontName='ResearchSans'
    style.fontSize=9
    style.leading=13
    rows=[[Paragraph(escape(str(c)),style) for c in frame.columns]]
    for _,row in frame.iterrows():
        rows.append([Paragraph(escape(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v)),style) for v in row])
    t=Table(rows,colWidths=[6.6*inch/len(frame.columns)]*len(frame.columns),repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EFF3')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),
        ('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#D7E0E5'))]))
    return t


def pdf(path,title,subtitle,pages,mode):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ResearchBody',fontName='ResearchSans',fontSize=10.5,leading=17,spaceAfter=16,textColor=colors.HexColor('#17324D')))
    styles['Title'].fontName='ResearchSansBold';styles['Heading1'].fontName='ResearchSansBold'
    styles['Title'].fontSize=25;styles['Title'].leading=31;styles['Title'].textColor=colors.HexColor('#17324D')
    styles['Heading1'].textColor=colors.HexColor('#007F82')
    story=[]
    for n,(heading,paragraphs,objects) in enumerate(pages):
        if n:story.append(PageBreak())
        if n==0:story.extend([Paragraph(escape(title),styles['Title']),Spacer(1,16),Paragraph(escape(subtitle),styles['ResearchBody']),Spacer(1,14)])
        story.append(Paragraph(escape(heading),styles['Heading1']))
        for p in paragraphs:story.append(Paragraph(escape(p),styles['ResearchBody']))
        for item in objects:
            story.append(Spacer(1,8))
            story.append(Image(str(item),width=6.55*inch,height=3.5*inch) if isinstance(item,Path) else table(item))
    def footer(canvas,doc):
        canvas.setFont('ResearchSans',7);canvas.setFillColor(colors.HexColor('#17324D'))
        canvas.drawString(.75*inch,.42*inch,f'QUANT RESEARCH STACK | {mode.upper()} DATA | v0.1.0')
        canvas.drawRightString(7.85*inch,.42*inch,str(doc.page))
        canvas.setStrokeColor(colors.HexColor('#CFDDE4'));canvas.line(.75*inch,.65*inch,7.85*inch,.65*inch)
    SimpleDocTemplate(str(path),pagesize=(8.5*inch,11*inch),rightMargin=.75*inch,leftMargin=.75*inch,
        topMargin=.65*inch,bottomMargin=.85*inch,title=title,author='Quant Research Stack').build(story,onFirstPage=footer,onLaterPages=footer)


def markdown(path,title,pages):
    lines=['# '+title,'','REAL OBSERVATIONS ONLY. Retrospective research; see limitations.','']
    for heading,paragraphs,objects in pages:
        lines.extend(['## '+heading,'',*sum(([p,''] for p in paragraphs),[])])
        for obj in objects:
            if isinstance(obj,Path):lines.extend([f'![{obj.stem}]({obj.name})',''])
            else:
                lines+=['| '+' | '.join(map(str,obj.columns))+' |','| '+' | '.join(['---']*len(obj.columns))+' |']
                for _,row in obj.iterrows():lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
                lines.append('')
    path.write_text('\n'.join(lines)+'\n')


def report(out,audit,performance,scores,ledgers,monthly,panel):
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    equity=out/'equity_curves.png';drawdown=out/'drawdown.png';hiringchart=out/'hiring_index.png'
    fig,ax=plt.subplots(figsize=(10,5.3))
    for name in ['composite','composite_long_only','eligible_equal_weight','spy_buy_hold']:
        r=ledgers[name].loc['2023':].net_return
        ax.plot(r.index,(1+r).cumprod(),label=name.replace('_',' '))
    ax.set(title='Observed SaaS equities: chronological test, Jan 2023-Aug 2026',ylabel='Compounded NAV after modeled costs')
    ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(equity,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.3))
    for name in ['composite','composite_long_only']:
        v=(1+ledgers[name].loc['2023':].net_return).cumprod();ax.plot(v.index,v/v.cummax().clip(lower=1)-1,label=name)
    ax.set(title='Actual-return portfolio drawdowns',ylabel='Drawdown');ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(drawdown,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.3));s=monthly.loc[:'2026-08-31'];ax.plot(s.index,s,color='#007F82')
    ax.axhline(100,ls='--',color='gray');ax.set(title='Indeed software-development postings: current historical vintage',ylabel='Index, February 1 2020 = 100')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(hiringchart,dpi=160);plt.close(fig)
    p=performance[performance.split=='test'];cols=['strategy','annualized_return','sharpe','max_drawdown']
    core=p[p.strategy.isin(['composite','composite_long_only','eligible_equal_weight','spy_buy_hold'])][cols]
    source='Sources: Yahoo Finance chart endpoint for adjusted monthly prices; SEC EDGAR Company Facts for original filed accounting observations; Indeed Hiring Lab via FRED for the software-development postings index.'
    caveat='The universe consists of 12 selected surviving SaaS companies. It is not the S&P 500 and is not survivorship-bias-free. Latest-vintage price adjustments and revised hiring history prevent a claim of fully vintage-correct out-of-sample performance.'
    factors=[
    ('01 / Abstract and research question',[
        'Can a fixed combination of momentum, low volatility, reported profitability and revenue growth rank subsequent returns within a small observed SaaS universe?',
        f"This run uses {audit['factor_rows']:,} eligible feature rows and {audit['quarterly_growth_observations']:,} SEC-derived quarterly growth observations. Every input is a retrieved observation or an explicitly documented calculation from observations. No generated proxy or generated fallback is used.",
        'The implementation runs on actual prices through August 2026. Historical valuation inputs were not constructed, so this four-signal study omits value rather than substituting invented values.',caveat],[]),
    ('02 / Sample and provenance',[
        'Companies: '+', '.join(audit['real_tickers'])+'. Price history starts in January 2015 or at the company IPO. Stocks enter only after the required observed price history and filed fundamentals are available.',
        f"The first portfolio date with six eligible names is {audit['first_portfolio_date']}. Development runs through December 2022; the chronological test is January 2023-August 2026.",
        source,
        'The normalized inputs and their hashes are included. The downloader retains original response hashes. Fundamental rows retain revenue tags, filing accession numbers, prior-year accessions and quarter-derivation flags.'
    ],[]),
    ('03 / SEC quarter construction',[
        'Revenue and operating income are taken from USD facts in 10-Q/10-K filings. Direct quarterly durations must span 70-110 days. When needed, Q4 is annual cumulative revenue minus the most recently available nine-month cumulative value with the same fiscal start.',
        'Q4 subtraction can mix filing vintages when comparative nine-month values are absent from the annual filing. Each derivation is flagged and both accessions are retained; these rows deserve manual review before stronger economic claims.',
        'Revenue growth compares the quarter with an observed quarter ending 330-400 days earlier. Operating margin requires operating income with the same start/end and available by the revenue filing date. Missing margins remain missing.',
        'The first reported quarter is retained. Availability is filing date plus one calendar day and joins require availability strictly before the rebalance date. This is a conservative date-level convention, not precise intraday filing reconstruction.'
    ],[]),
    ('04 / Signals and portfolio design',[
        'Momentum = adjusted close at t-1 divided by adjusted close at t-12 minus one. Low volatility = negative 12-month return standard deviation through t-1. Profitability = most recently known operating margin. Growth = most recently known year-over-year quarterly revenue growth.',
        'Signals are winsorized at cross-sectional 5th/95th percentiles and standardized within each date. Equal weights form the composite; all four values must be observed. The universe is small and tech-focused, so there is no claim of broad sector neutrality.',
        'With at least six usable names, upper/lower terciles form +50% long/-50% short books. A separate long-only portfolio holds the upper tercile at 100%. An eligible-universe equal-weight portfolio and SPY provide comparisons.'
    ],[]),
    ('05 / Backtesting framework',[
        'At each month-end, previous positions first earn observed adjusted-price returns. Cash interest and borrow are accounted for, then new targets are traded. Price-derived signals lag one full observation; the target never earns the return ending on its own row.',
        'The engine solves fees against after-fee NAV, tracks drifting holdings and cash, and rejects missing held returns. Base commission plus slippage is 10 basis points per dollar bought or sold. Borrow is 100 basis points annually, accrued monthly. Cash earns zero.',
        'Gross exposure is 100% for the long-short strategy, with zero net exposure. Turnover counts both purchases and sales. No terminal liquidation, realistic borrow locates, market-impact model, intramonth margin calls, taxes or delisted-company sample is included.',
        'Sharpe/Sortino use a zero return hurdle. Benchmark beta and arithmetic alpha are diagnostics, not proof that all risk exposures were controlled. Adjusted-close changes are a vendor-adjusted approximation to total returns.'
    ],[]),
    ('06 / Measured chronological test results',[
        'All table entries are calculated from the retrieved prices. Annualized returns and drawdowns are decimal fractions. No parameter or seed search was performed to make the outcome favorable.',
        'Long-only SaaS, dollar-neutral SaaS and SPY have different risks. Compare the long-only composite with the eligible-universe benchmark as well as SPY. A realized return advantage is not automatically stock-selection alpha.',caveat
    ],[core]),
    ('07 / Return paths',[
        'Curves compound net returns in the test segment, carrying the strategy positions through the development/test boundary. They are historical calculations under execution assumptions, not a live track record.',
        'Future-data perturbation tests verify that later prices and filings cannot change earlier calculated features. They cannot remove the surviving-company selection bias or the source provider revision problem.'
    ],[equity]),
    ('08 / Risk and transaction costs',[
        'Drawdown includes the initial NAV peak. The cost-sensitivity table reruns identical targets at 0/5/10/25/50 one-way basis points, holding short borrowing at 100 basis points annually.',
        'Inspect the exported monthly holdings, ledgers, rank ICs and signal correlation matrix. A concentrated SaaS sample can have substantial common growth and duration exposures even with zero net dollar exposure.'
    ],[drawdown]),
    ('09 / Single signals and robustness',[
        'The same backtester evaluates every standalone signal. The comparison is exploratory; four signals and multiple outcomes create a multiple-testing problem. No corrected significance claim is made.',
        'The 2023 split is a fixed chronological evaluation, but it was chosen in a contemporary retrospective project. Universe selection, specification choices and current data vintages mean it is not an untouched prospective experiment.'
    ],[p[p.strategy.str.endswith('_z')][cols]]),
    ('10 / Conclusion, reproduction and sources',[
        'This is now a real-observation study with a smaller scope. It supports discussion of data engineering, financial definitions, implementation and the measured results. It does not establish a broad-market investment edge.',
        'Reproduce: python -m real_research.analyze. Refresh sources: python -m real_research.download --refresh. Tests: python -m unittest discover -s tests -v.',
        'SEC source documentation: https://www.sec.gov/search-filings/edgar-application-programming-interfaces. Yahoo endpoints and per-company source hashes are listed in data/real/manifest.json.',
        'Potential next work: historical constituents and delistings, actual valuation denominators, source vintages, sector controls and a prospectively reserved test. None are silently substituted in this version.'
    ],[])]
    hiring=[
    ('01 / Question and actual data',[
        'Does the broader software-hiring environment improve forecasts of reported SaaS revenue growth, or help explain subsequent equity returns?',
        f"The study uses {audit['hiring_observations']:,} actual daily index observations from Indeed Hiring Lab, distributed by FRED as IHLIDXUSTPSOFTDEVE. The index begins February 2020 and is sampled at month-end.",
        'This is an occupation-wide hiring index, not a history of job openings at HubSpot, Salesforce or any other individual company. The former company-specific hiring claims are not supported by this dataset.',
        'Indeed revises historical values and changed its seasonal-adjustment methodology. The analysis is explicitly retrospective on the retrieved vintage; lagging the index does not recover what was available at a past date.'
    ],[]),
    ('02 / Hiring features',[
        'Hiring growth is index(t-1) / index(t-4) - 1. Acceleration subtracts hiring growth from three months earlier. The one-month lag reduces same-month timing assumptions, but does not resolve revision bias.',
        'A common occupational signal varies over time, not across companies. It can represent a hiring backdrop in a pooled forecasting model, but it cannot rank firms by their own hiring activity.',
        'Missing observations are not filled with invented counts. There is no LinkedIn scraping claim, no company job-count extrapolation and no generated fallback.'
    ],[hiringchart]),
    ('03 / Revenue forecast comparison',[
        'The baseline ridge model uses latest reported revenue growth, operating margin, price momentum and company indicators. The augmented model adds aggregate hiring growth and acceleration. Ridge penalty is fixed at 1; scaling is estimated on training rows only.',
        'For each monthly feature date, the target is the next observed fiscal quarter ending after that date and within 110 days. A label enters training only after its SEC reporting availability date. Repeated ticker/quarter targets are deduplicated in training.',
        'Models expand from 2022 with at least 50 matured rows and eight target-period dates. Test begins January 2023. Scoring uses one latest forecast per company/quarter/model and only labels available by August 31, 2026.',
        'RMSE and MAE use revenue-growth decimal units. The actual independent macro sample is much smaller than the company-row count because all companies share the hiring index.'
    ],[scores]),
    ('04 / Equity-return tests',[
        'Return regressions compare the equal-weight available-company basket with SPY at 1-, 3- and 6-month horizons. Hiring growth and lagged SPY momentum are predictors. HAC covariance uses at least three lags and at least the return horizon.',
        'The basket is descriptive and reweights to available real observations; it inherits selection bias. The small post-2020 macro sample, pandemic effects, revised history and overlapping returns restrict inference. Coefficients and t-statistics are exploratory.',
        'A separate simple policy holds SPY when lagged hiring growth is positive and cash otherwise. This tests aggregate market timing, not company-specific hiring alpha. It uses the same trading-cost engine.'
    ],[p[p.strategy.isin(['hiring_market_timing','spy_buy_hold'])][cols]]),
    ('05 / Interpretation and limitations',[
        'A lower forecasting error would be evidence of descriptive incremental fit in this sample, not causal evidence that hiring drives revenue. A higher error would be equally reportable. Neither outcome is altered or omitted.',
        'The narrow occupational index and SaaS company sample do not represent each employer, and may jointly respond to growth expectations, financing conditions or recession fears. Models may capture shared macro conditions rather than a new investment signal.',
        'SEC first-report labels reduce accounting look-ahead, but current Indeed history is not an original-vintage dataset. A defensible prospective test must archive vintages or begin collecting observations before the evaluation period.',
        'The former synthetic company hiring panel and result graphics are removed from the current repository tree. Their history is retained in Git; prior numeric claims should not be used as empirical results.'
    ],[]),
    ('06 / Reproduce and audit',[
        'Run python -m real_research.analyze from either repository. Review revenue_predictions.csv, forecast_scores.csv, hiring_return_observations.csv and the 1m/3m/6m regression files. The audit includes exact normalized source hashes.',
        'Series: https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE. Attribution: Indeed, Software Development Job Postings on Indeed in the United States, retrieved via FRED, Federal Reserve Bank of St. Louis.',
        'Methodology and CC BY 4.0 data license: https://github.com/hiring-lab/job_postings_tracker. The source explicitly documents historical revisions. Its index is not an indicator of Indeed or Recruit Holdings revenue.',
        'Company-specific hiring research remains untested until real firm-level historical snapshots are obtained. The implemented project now answers the narrower aggregate-hiring question honestly.'
    ],[])]
    pdf(out/'factor_research.pdf','Observed SaaS Equity Factor Research','Real Yahoo prices and SEC filings | data through August 2026',factors,'historical')
    pdf(out/'hiring_research.pdf','Software Hiring and SaaS Growth','Real Indeed/FRED index and SEC revenue | retrospective study',hiring,'historical')
    markdown(out/'factor_research.md','Observed SaaS Equity Factor Research',factors)
    markdown(out/'hiring_research.md','Software Hiring and SaaS Growth',hiring)
