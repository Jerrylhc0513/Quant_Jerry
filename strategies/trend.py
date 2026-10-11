"""
trend.py - hold an index ETF only while it is in an uptrend; otherwise sit in cash.

  spy_trend   SPY while it closes above its 200-day average, cash otherwise
  qqq_trend   the same for QQQ (Nasdaq-100, tech heavy)

The aim is not a higher return than buy-and-hold in rising markets, but avoiding most of the big falls
(2020, 2022). Cash earns nothing in the engine, so the backtest slightly understates these strategies.
"""
from engine import Order, Strategy


class Trend(Strategy):
    def __init__(self, name, title, etf, window=200):
        self.name, self.title, self.etf, self.window = name, title, etf, window
        self.description = f"Hold {etf} while it closes above its {window}-day average, cash otherwise"
        self.cost_bps = 2.0
        self.start = "2017-01-03"

    def prepare(self, md):
        c = md.C[self.etf].ffill() if self.etf in md.C else None
        self.up = None if c is None else (c > c.rolling(self.window, min_periods=int(self.window * 0.75)).mean())

    def decide(self, i, book, md):
        if self.up is None:
            return []
        want = bool(self.up.iloc[i])
        have = self.etf in book.positions
        if want and not have and not book.pending:
            return [Order(self.etf, weight=1.0, reason=f"above {self.window}-day average")]
        if have and not want:
            return [Order(self.etf, weight=0.0, reason=f"below {self.window}-day average")]
        return []


SPY_TREND = Trend("spy_trend", "SPY with a 200-day trend filter", "SPY")
QQQ_TREND = Trend("qqq_trend", "QQQ with a 200-day trend filter", "QQQ")
