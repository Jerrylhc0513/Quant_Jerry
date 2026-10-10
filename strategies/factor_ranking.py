"""
factor_ranking.py - the multi-factor ranking strategy (model.py scores) as engine strategies.

Every `rebalance` trading days: hold the best-ranked stocks, keep a holding while its rank stays inside
`hold_rank`, fill free slots with the best new names (same rules as backtest.PROFILES).
Uses data/scores/<score>.parquet, so run model.py first (run_daily.bat does).
"""
import numpy as np
import pandas as pd

import backtest as B
import fundamentals as FU
import model as M
from engine import Order, Strategy


class FactorRanking(Strategy):
    def __init__(self, name, title, profile: str, score: str = "wf", description=""):
        self.name, self.title, self.description = name, title, description
        self.prof = B.PROFILES[profile]
        self.score = score
        self.cost_bps = B.HALF_SPREAD_BPS + B.COMMISSION_BPS

    def prepare(self, md):
        sc = M.load_scores(self.score)
        S = sc.pivot(index="date", columns="symbol", values="score").sort_index()
        self.S = S.reindex(index=md.dates)
        self.has = set(S.index)
        R = md.C.pct_change(fill_method=None)
        self.vol = R.rolling(B.VOL_WINDOW, min_periods=40).std()
        sec = FU.load_sectors()
        self.sectors = sec if len(sec) else None

    def first_day(self, md):
        first = min(self.has) if self.has else md.dates[-1]
        return int(md.dates.searchsorted(first))

    def decide(self, i, book, md):
        d = md.dates[i]
        if d not in self.has:
            return []
        last = book.state.get("last_rebalance")          # stored as a date: indexes shift between runs
        if last is not None and i - int(md.dates.searchsorted(pd.Timestamp(last))) < self.prof.rebalance:
            return []
        score = self.S.iloc[i].dropna()
        if score.empty:
            return []
        book.state["last_rebalance"] = str(d.date())
        held = list(book.positions)
        new_hold = B.select_portfolio(score, held, self.prof.top_n, self.prof.hold_rank,
                                      self.sectors, self.prof.max_per_sector)
        cur = pd.Series(book.weights(), dtype=float)
        tw = B.plan_weights(new_hold, cur, self.vol.iloc[i], self.prof, first=not held)
        orders = [Order(s, weight=0.0, reason="dropped out of the ranking") for s in held if s not in new_hold]
        orders += [Order(s, weight=float(w), reason="ranking") for s, w in tw.items()
                   if abs(float(w) - cur.get(s, 0.0)) > 1e-4]
        return orders


WEEKLY_25 = FactorRanking("factor_weekly25", "Factor ranking, 25 stocks weekly", "model",
                          description="model.py scores; 25 stocks, rebalance every 5 days, inverse-vol weights")
MONTHLY_5 = FactorRanking("factor_monthly5", "Factor ranking, 5 stocks monthly", "real",
                          description="model.py scores; 5 stocks, rebalance every 21 days, equal weights")
