"""Tests use retrieved observations, with deliberate perturbations for leakage checks."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from real_research.analyze import load,features,targets,forecasts
from real_research.engine import run,Costs

ROOT=Path(__file__).resolve().parents[1]

class RealResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.f,cls.h,cls.m=load(ROOT/'data/real')
        cls.panel,cls.r,cls.monthly=features(cls.p,cls.f,cls.h)

    def test_only_historical_manifest(self):
        self.assertEqual(self.m['data_mode'],'historical')
        self.assertEqual(set(self.p.ticker)-{'SPY'},set(self.m['universe']))
        self.assertFalse(self.p.ticker.str.startswith('SIM').any())

    def test_real_source_observation_count(self):
        self.assertGreater(len(self.h),1000)
        self.assertGreater(len(self.f),300)
        self.assertEqual(self.h.hiring_index.iloc[0],100.)
        self.assertEqual(str(self.h.date.iloc[0].date()),'2020-02-01')

    def test_fundamental_availability(self):
        self.assertTrue((self.panel.fundamental_available_at<self.panel.date).all())

    def test_future_prices_do_not_change_earlier_features(self):
        p=self.p.copy();p.loc[p.date>='2024','adjusted_close']*=2
        later,_,_=features(p,self.f,self.h)
        pd.testing.assert_frame_equal(self.panel[self.panel.date<'2024'].reset_index(drop=True),later[later.date<'2024'].reset_index(drop=True))

    def test_future_filings_do_not_change_earlier_features(self):
        f=self.f.copy();f.loc[f.available_at>='2024','operating_margin']+=1
        later,_,_=features(self.p,f,self.h)
        pd.testing.assert_frame_equal(self.panel[self.panel.date<'2024'].reset_index(drop=True),later[later.date<'2024'].reset_index(drop=True))

    def test_zero_net_and_gross_limit(self):
        w=targets(self.panel,self.r,'composite')
        np.testing.assert_allclose(w.sum(axis=1),0,atol=1e-12)
        self.assertLessEqual(w.abs().sum(axis=1).max(),1+1e-12)

    def test_no_same_row_return(self):
        r=self.r[['SPY']].dropna().iloc[:3]
        w=pd.DataFrame(1.,index=r.index,columns=r.columns)
        bt=run(r,w,Costs(0,0,0))
        self.assertAlmostEqual(bt.ledger.nav.iloc[0],1.)
        self.assertAlmostEqual(bt.ledger.nav.iloc[1],1+r.SPY.iloc[1])

    def test_cash_cost_identity_on_actual_returns(self):
        r=self.r.loc['2023':];panel=self.panel[self.panel.date>='2023']
        bt=run(r,targets(panel,r,'composite'))
        np.testing.assert_allclose(bt.ledger.net_return,bt.ledger.gross_return-bt.ledger.trading_cost-bt.ledger.borrow_cost,atol=1e-12)
        np.testing.assert_allclose(bt.weights.sum(axis=1)+bt.ledger.cash_weight,1,atol=1e-12)

    def test_missing_held_price_stops(self):
        r=self.r[['SPY']].dropna().iloc[:3].copy();r.iloc[1,0]=np.nan
        w=pd.DataFrame(1.,index=r.index,columns=r.columns)
        with self.assertRaisesRegex(ValueError,'Missing return'):run(r,w)

    def test_matured_label_forecasts(self):
        saved=pd.read_csv(ROOT/'reports/revenue_predictions.csv',parse_dates=['date','max_training_label'])
        self.assertTrue((saved.max_training_label<saved.date).all())
        self.assertTrue(np.isfinite(saved.prediction).all())

    def test_reported_target_identity(self):
        saved=pd.read_csv(ROOT/'reports/revenue_predictions.csv',parse_dates=['target_period'])
        joined=saved.merge(self.f[['ticker','period_end','revenue_growth']],left_on=['ticker','target_period'],right_on=['ticker','period_end'])
        np.testing.assert_allclose(joined.target,joined.revenue_growth,atol=1e-12)

    def test_reproducible_features(self):
        saved=pd.read_csv(ROOT/'reports/features.csv',parse_dates=['date'])
        actual=self.panel.merge(saved[['date','ticker','composite']],on=['date','ticker'],suffixes=('_new','_saved'))
        np.testing.assert_allclose(actual.composite_new,actual.composite_saved,equal_nan=True,atol=1e-12)

if __name__=='__main__':unittest.main()
