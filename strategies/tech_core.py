"""
tech_core.py - hold the TechDist basket (td_data.POOL_A, 10 large tech / AI stocks) and only manage RISK.

TechDist showed that the direction of these stocks over 1-5 days could not be predicted, but their
volatility can. So these strategies never try to pick winners or time tops. They always hold the
same basket and only change HOW MUCH is held:

  tech10_hold       equal weights bought on the first day, never touched (the benchmark)
  tech10_ew         equal weights, rebalanced back to equal every 21 trading days
  tech10_trend      tech10_ew, but only 50% invested while QQQ closes below its 200-day average
  tech10_vol        weights proportional to 1 / volatility, and the whole basket scaled so its predicted
                    volatility is at most 20% a year (cash makes up the rest, no leverage)
  tech10_vol_trend  tech10_vol plus the QQQ trend switch

Volatility: exponentially weighted (half-life ~11 days) daily volatility of each stock, and the 60-day
covariance of the basket for the portfolio level. Only data up to the signal day is used.
The parameters (20% target, 50% floor, weekly check, 2% band) were fixed before any backtest and are
not tuned on results. Signals start in 2020 (the same era as TechDist).
Note: the basket was chosen today (hindsight), so judge the risk-managed versions against tech10_hold
and tech10_ew, not against SPY.
"""
import numpy as np
import pandas as pd

from engine import Order, Strategy
from strategies.buy_hold import BuyAndHold
from td_data import POOL_A

START = "2020-01-02"


class TechCore(Strategy):
    def __init__(self, name, title, description, vol_target=None, trend=False, floor=0.5,
                 rebalance=5, band=0.02, buffer=0.0, whole=False):
        self.name, self.title, self.description = name, title, description
        self.vol_target, self.trend, self.floor = vol_target, trend, floor
        self.rebalance, self.band, self.buffer = rebalance, band, buffer
        self.whole = whole          # True: one stock off by > band re-balances every stock (stays invested)
        self.cost_bps = 5.0
        self.start = START

    def prepare(self, md):
        self.syms = [s for s in POOL_A if s in md.C.columns]
        C = md.C[self.syms].astype("float64")
        self.R = np.log(C).diff()
        self.sig = self.R.ewm(halflife=11, min_periods=20).std() * np.sqrt(252)
        q = md.C["QQQ"].astype("float64") if "QQQ" in md.C and md.C["QQQ"].notna().sum() > 200 else None
        if q is None:                                  # benchmark file without QQQ: use the TechDist copy
            try:
                import td_data as T
                d = T.load_daily(["QQQ"])
                q = d.set_index("date")["adj_close"].reindex(md.dates)
            except (OSError, KeyError):
                q = None
        if q is None or q.notna().sum() < 200:
            if self.trend:
                raise SystemExit("no QQQ prices for the trend switch: run python data.py update")
            self.up = np.ones(len(md.dates), dtype=bool)
        else:
            q = q.ffill()
            ma = q.rolling(200, min_periods=150).mean()
            if self.buffer <= 0:
                self.up = (q > ma).to_numpy()
            else:                                      # hysteresis: switch only on a clear break
                qa, ma_ = q.to_numpy(), ma.to_numpy()
                up = np.ones(len(qa), dtype=bool)
                state = True
                for t in range(len(qa)):
                    if np.isfinite(qa[t]) and np.isfinite(ma_[t]):
                        if state and qa[t] < ma_[t] * (1 - self.buffer):
                            state = False
                        elif not state and qa[t] > ma_[t] * (1 + self.buffer):
                            state = True
                    up[t] = state
                self.up = up
        self.Cn = C.to_numpy()

    def target(self, i) -> dict:
        ok = [s for k, s in enumerate(self.syms) if np.isfinite(self.Cn[i, k])]
        if not ok:
            return {}
        if self.vol_target is None:
            w = pd.Series(1.0 / len(ok), index=ok)
        else:
            sig = self.sig.iloc[i][ok]
            sig = sig.fillna(sig.median() if sig.notna().any() else 0.4).clip(lower=0.05)
            w = (1 / sig) / (1 / sig).sum()
            R = self.R.iloc[max(0, i - 59): i + 1][ok].fillna(0.0)
            cov = R.cov().to_numpy() * 252
            port_vol = float(np.sqrt(w.to_numpy() @ cov @ w.to_numpy())) if len(R) >= 20 else np.nan
            if np.isfinite(port_vol) and port_vol > 0:
                w *= min(1.0, self.vol_target / port_vol)
        if self.trend and not self.up[i]:
            w *= self.floor
        return w.to_dict()

    def decide(self, i, book, md):
        regime = bool(self.up[i])
        last = book.state.get("last_check")
        switched = self.trend and book.state.get("regime") is not None and book.state["regime"] != regime
        due = last is None or i - int(md.dates.searchsorted(pd.Timestamp(last))) >= self.rebalance
        if not (due or switched):
            return []
        book.state["last_check"] = str(md.dates[i].date())
        book.state["regime"] = regime
        tgt = self.target(i)
        cur = book.weights()
        orders = [Order(s, weight=0.0, reason="left the basket") for s in cur if s not in tgt]
        off = {s: abs(w - cur.get(s, 0.0)) for s, w in tgt.items()}
        go_all = self.whole and max(off.values(), default=0.0) > self.band
        for s, w in tgt.items():
            if (off[s] > 0.005 if go_all else off[s] > self.band) or (switched and off[s] > 1e-4):
                if switched:
                    why = "risk-on (QQQ back above 200-day avg)" if regime else "risk-off (QQQ below 200-day avg)"
                else:
                    why = "rebalance" + (" while risk-off" if self.trend and not regime else "")
                orders.append(Order(s, weight=w, reason=why))
        return orders


TECH10_HOLD = BuyAndHold("tech10_hold", "Tech basket buy & hold", {s: 1 / len(POOL_A) for s in POOL_A},
                         "10 large tech / AI stocks, equal weights on the first day, never traded", start=START)
TECH10_EW = TechCore("tech10_ew", "Tech basket, equal weight monthly",
                     "10 large tech / AI stocks, back to equal weights every 21 trading days", rebalance=21, band=0.0)
TECH10_TREND = TechCore("tech10_trend", "Tech basket + QQQ trend switch",
                        "Equal weights; only 50% invested while QQQ is below its 200-day average",
                        trend=True, rebalance=21, band=0.0)
TECH10_VOL = TechCore("tech10_vol", "Tech basket, volatility-managed",
                      "1/vol weights, basket scaled to at most 20% predicted annual volatility, weekly check",
                      vol_target=0.20)
TECH10_VOL_TREND = TechCore("tech10_vol_trend", "Tech basket, volatility-managed + trend switch",
                            "tech10_vol, and only half of that while QQQ is below its 200-day average",
                            vol_target=0.20, trend=True)
# Added after seeing tech10_trend's order list (2020-2026: 24 switches, about 18 of them reversed within
# 1-3 days, plus many orders of a few dollars). Same idea, two fixed changes: the regime only flips when
# QQQ closes more than 2% beyond its 200-day average, and a stock is only traded when its weight is more
# than 2 percentage points away from target (then every stock off by more than 0.5 points is re-balanced). Judge it with that history in mind (one round of changes).
TECH10_TREND_BUF = TechCore("tech10_trend_buf", "Tech basket + QQQ trend switch with a 2% buffer",
                            "Equal weights; 50% invested after QQQ closes 2% below its 200-day average, back to "
                            "100% after it closes 2% above; only trades weights off by more than 2 points",
                            trend=True, rebalance=21, band=0.02, buffer=0.02, whole=True)
