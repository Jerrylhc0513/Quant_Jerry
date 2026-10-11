"""
tech.py - technology stocks only. The fair benchmark here is QQQ (or XLK), not SPY: tech as a whole did very
well since 2016, so any basket of tech stocks tends to beat SPY. The question is whether picking WITHIN tech
beats simply holding QQQ.

  qqq_hold          QQQ bought once and held (the benchmark)
  tech_momentum10   liquid tech stocks; every 21 trading days hold the 10 with the best 12-1 month return
                    (a holding is kept while it stays in the top 20), equal weights
  tech_factor10     the model.py score (data/scores/wf.parquet) restricted to tech stocks; top 10, same rules

Tech = SEC SIC codes mapped to the "Technology" sector (software, semiconductors, computers, electronics,
internet services such as GOOGL and META). AMZN and TSLA are classed as consumer stocks and are not included.
"""
import numpy as np
import pandas as pd

import data as D
import fundamentals as FU
import model as M
from engine import Order, Strategy
from strategies.buy_hold import BuyAndHold

TECH = FU.SECTORS.index("Technology")


def tech_mask(md, min_price=5.0, min_adv=20e6) -> pd.DataFrame:
    sec = FU.load_sectors()
    cols = [c for c in md.C.columns if c not in D.BENCHMARKS and sec.get(c, -1) == TECH]
    return (md.P[cols] >= min_price) & (md.ADV[cols] >= min_adv)


class TechRanking(Strategy):
    def __init__(self, name, title, description, top_n=10, keep_rank=20, rebalance=21):
        self.name, self.title, self.description = name, title, description
        self.top_n, self.keep_rank, self.rebalance = top_n, keep_rank, rebalance
        self.cost_bps = 5.0
        self.start = "2017-01-03"

    def scores(self, md) -> pd.DataFrame:
        raise NotImplementedError

    def prepare(self, md):
        mask = tech_mask(md)
        self.S = self.scores(md).reindex(index=md.dates, columns=mask.columns).where(mask)

    def decide(self, i, book, md):
        last = book.state.get("last_rebalance")
        if last is not None and i - int(md.dates.searchsorted(pd.Timestamp(last))) < self.rebalance:
            return []
        s = self.S.iloc[i].dropna().sort_values(ascending=False)
        if len(s) < self.top_n:
            return []
        book.state["last_rebalance"] = str(md.dates[i].date())
        rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
        held = list(book.positions)
        keep = [x for x in held if rank.get(x, 10 ** 6) <= self.keep_rank][: self.top_n]
        new = [x for x in s.index if x not in keep][: self.top_n - len(keep)]
        target = keep + new
        w = 1.0 / len(target)
        orders = [Order(x, weight=0.0, reason="fell out of the top") for x in held if x not in target]
        orders += [Order(x, weight=w, reason=f"rank {int(rank[x])}") for x in target]
        return orders


class TechMomentum(TechRanking):
    def scores(self, md):
        C = md.C
        return C.shift(21) / C.shift(252) - 1                     # 12-month return skipping the last month


class TechFactor(TechRanking):
    def scores(self, md):
        sc = M.load_scores("wf")
        return sc.pivot(index="date", columns="symbol", values="score")


QQQ_HOLD = BuyAndHold("qqq_hold", "QQQ buy & hold", {"QQQ": 1.0}, "100% QQQ (Nasdaq-100), never sold. Tech benchmark.")
TECH_MOMENTUM = TechMomentum("tech_momentum10", "Tech momentum, 10 stocks monthly",
                             "Liquid tech stocks; hold the 10 best 12-1 month returns, rebalance every 21 days")
TECH_FACTOR = TechFactor("tech_factor10", "Tech factor score, 10 stocks monthly",
                         "model.py wf score within liquid tech stocks; top 10, rebalance every 21 days")
