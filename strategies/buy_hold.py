"""
buy_hold.py - buy once and hold: the simplest strategy, and the template for writing your own.

  spy_hold        100% SPY. The benchmark every other idea has to beat.
  nvda_googl_hold half NVDA, half GOOGL, bought on the first day and never sold.
                  Careful: picking these two today uses hindsight (we already know they did well since 2016),
                  so its backtest is optimistic. Its live paper account from today on is a fair test.
"""
from engine import Order, Strategy


class BuyAndHold(Strategy):
    def __init__(self, name, title, weights: dict, description="", start="2017-01-03"):
        self.name, self.title, self.weights, self.description, self.start = name, title, weights, description, start
        self.cost_bps = 2.0

    def decide(self, i, book, md):
        if book.positions or book.pending:
            return []                              # already invested: hold forever
        return [Order(s, weight=w, reason="initial buy") for s, w in self.weights.items()]


SPY_HOLD = BuyAndHold("spy_hold", "SPY buy & hold", {"SPY": 1.0}, "100% SPY, never sold. The benchmark.")
NVDA_GOOGL = BuyAndHold("nvda_googl_hold", "NVDA + GOOGL buy & hold", {"NVDA": 0.5, "GOOGL": 0.5},
                        "Half NVDA, half GOOGL, never sold. Backtest has hindsight bias; the live record is fair.")
