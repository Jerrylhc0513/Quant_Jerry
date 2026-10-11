"""
engine.py - one execution engine for every strategy in strategies/

A strategy only decides WHAT to trade. After the close of each day the engine calls
    strategy.decide(i, book, md) -> list[Order]
and the engine handles everything else, identically in backtests and in the live paper accounts:
  - orders fill at the next day's open, cost_bps charged on each side
  - an order sets a target weight of the account (0 = sell everything), or spends a fraction of free cash
  - optional exit lines attached to a buy: take-profit tp (+50% = 0.5), stop-loss sl (0.2 = -20%),
    max_days (sell at the close after this many trading days). They are checked every day with the day's
    high and low: stop first if both are touched (conservative); a gap through a line fills at the open.
  - a stock whose data stops (delisting) is sold at its last value after MISSING_DAYS days without prices

Positions are tracked as values and growth factors (ratios of adjusted prices on consecutive days), so stored
live accounts stay correct when adjusted prices are rebased after new splits or dividends.

Writing a strategy (see strategies/buy_hold.py for the smallest example):
    from engine import Strategy, Order
    class MyIdea(Strategy):
        name = "my_idea"; title = "My idea"; description = "..."
        def prepare(self, md):            # optional: precompute signals for all days (no look-ahead!)
            ...
        def decide(self, i, book, md):    # after the close of day i
            return [Order("NVDA", weight=0.5, tp=0.3, sl=0.1, max_days=60)]
    STRATEGY = MyIdea()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import data as D

MISSING_DAYS = 5
log = logging.getLogger("engine")


# ============================== Orders, positions, book ==============================
@dataclass
class Order:
    symbol: str
    weight: float | None = None       # target share of the account (0 = sell all)
    cash_frac: float | None = None    # buy with this fraction of the cash available at the open
    tp: float | None = None           # take-profit, e.g. 0.5 = +50% from entry
    sl: float | None = None           # stop-loss, e.g. 0.2 = -20% from entry
    max_days: int | None = None       # time exit after this many trading days
    reason: str = ""


@dataclass
class Position:
    symbol: str
    entry_date: str
    value: float                      # current value (marked at the last close)
    invested: float                   # cash paid in (including costs)
    g: float = 1.0                    # price growth since entry
    days: int = 0                     # trading days held
    tp: float | None = None
    sl: float | None = None
    max_days: int | None = None
    last_i: int = -1                  # index of the last price used (not stored across runs)
    missing: int = 0
    reason: str = ""


@dataclass
class Book:
    cash: float
    positions: dict = field(default_factory=dict)       # symbol -> Position
    pending: list = field(default_factory=list)         # orders for the next open
    state: dict = field(default_factory=dict)           # free storage for the strategy
    history: list = field(default_factory=list)         # daily records
    trades: list = field(default_factory=list)          # closed trades
    last_date: str | None = None

    @property
    def nav(self) -> float:
        return self.cash + sum(p.value for p in self.positions.values())

    def weights(self) -> dict:
        n = self.nav
        return {s: p.value / n for s, p in self.positions.items()} if n > 0 else {}

    # ---- persistence (live paper accounts) ----
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        st = {"cash": self.cash, "last_date": self.last_date, "state": self.state,
              "positions": [{k: v for k, v in asdict(p).items() if k != "last_i"} for p in self.positions.values()],
              "pending": [asdict(o) for o in self.pending]}
        (folder / "book.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        pd.DataFrame(self.history).to_csv(folder / "history.csv", index=False)
        pd.DataFrame(self.trades).to_csv(folder / "trades.csv", index=False)

    @classmethod
    def load(cls, folder: Path, capital: float) -> "Book":
        f = folder / "book.json"
        if not f.exists():
            return cls(cash=capital)
        st = json.loads(f.read_text(encoding="utf-8"))
        b = cls(cash=st["cash"], state=st.get("state", {}), last_date=st.get("last_date"))
        b.positions = {p["symbol"]: Position(**p) for p in st.get("positions", [])}
        b.pending = [Order(**o) for o in st.get("pending", [])]
        for name, attr in (("history.csv", "history"), ("trades.csv", "trades")):
            p = folder / name
            if p.exists() and p.stat().st_size > 0:
                try:
                    setattr(b, attr, pd.read_csv(p).to_dict("records"))
                except pd.errors.EmptyDataError:
                    pass
        return b


# ============================== Market data ==============================
class MarketData:
    """Adjusted open/high/low/close, raw close, volume and 60-day dollar volume for every stock (+ SPY),
    as date x symbol float32 frames. Loaded once and shared by all strategies."""

    def __init__(self, start=None, end=None):
        t0 = time.time()
        df = D.load_daily(start=start, end=end, adjusted=True,
                          columns=["open", "high", "low", "close", "volume", "ret"])
        if D.BENCHMARK_FILE.exists():
            b = D.load_benchmark("SPY")
            if start is not None:
                b = b[b["date"] >= pd.Timestamp(start)]
            if end is not None:
                b = b[b["date"] <= pd.Timestamp(end)]
            df = pd.concat([df[df["symbol"] != "SPY"], b[df.columns.intersection(b.columns)]], ignore_index=True)
        piv = lambda c: df.pivot(index="date", columns="symbol", values=c).sort_index().astype("float32")
        self.C = piv("adj_close")
        al = lambda x: x.reindex_like(self.C)
        self.O, self.H, self.L = al(piv("adj_open")), al(piv("adj_high")), al(piv("adj_low"))
        self.P = al(piv("close"))
        self.V = al(piv("volume"))
        del df
        self.ADV = (self.P * self.V).rolling(60, min_periods=50).mean()
        self.dates = self.C.index
        self.symbols = self.C.columns
        self.col = {s: k for k, s in enumerate(self.symbols)}
        self._np = {k: getattr(self, k).to_numpy() for k in ("O", "H", "L", "C")}
        log.info("Market data: %d days x %d symbols (%.0f s)", len(self.dates), len(self.symbols), time.time() - t0)

    def px(self, kind: str, i: int, sym: str) -> float:
        j = self.col.get(sym)
        return float("nan") if j is None else float(self._np[kind][i, j])


# ============================== Strategy base ==============================
class Strategy:
    name = "base"
    title = "Base"
    description = ""
    capital = 100_000.0
    cost_bps = 5.0
    start = None            # first signal date for backtests (None = as early as the strategy can)
    enabled = True

    def prepare(self, md: MarketData) -> None:
        """Precompute signals for all days. Values used on day i must only depend on data up to day i."""

    def first_day(self, md: MarketData) -> int:
        return 0 if self.start is None else int(md.dates.searchsorted(pd.Timestamp(self.start)))

    def decide(self, i: int, book: Book, md: MarketData) -> list[Order]:
        raise NotImplementedError


# ============================== Engine ==============================
def _close(book: Book, p: Position, value: float, date, why: str, c: float) -> None:
    proceeds = value * (1 - c)
    book.cash += proceeds
    book.trades.append({"symbol": p.symbol, "entry_date": p.entry_date, "exit_date": str(date.date()),
                        "days": p.days, "ret_net": proceeds / p.invested - 1 if p.invested > 0 else np.nan,
                        "exit": why, "entry_reason": p.reason})
    del book.positions[p.symbol]


def step(st: Strategy, md: MarketData, book: Book, i: int) -> None:
    """Run trading day i: open (fills), intraday exit lines, close (marks, time exits)."""
    c = st.cost_bps / 1e4
    date = md.dates[i]
    # ---- 1. open: mark positions to the open ----
    opens = {}
    for p in book.positions.values():
        o = md.px("O", i, p.symbol)
        prev = md.px("C", p.last_i, p.symbol) if p.last_i >= 0 else float("nan")
        if np.isfinite(o) and np.isfinite(prev) and prev > 0:
            p.value *= o / prev
            p.g *= o / prev
            opens[p.symbol] = o
        elif np.isfinite(o):
            opens[p.symbol] = o
    nav_open = book.nav
    # ---- 2. execute pending orders: sells / reductions first, then buys ----
    orders = book.pending
    book.pending = []
    sells = [o for o in orders if o.weight is not None and o.symbol in book.positions
             and o.weight * nav_open < book.positions[o.symbol].value]
    buys = [o for o in orders if o not in sells]
    for o in sells:
        p = book.positions[o.symbol]
        if o.symbol not in opens:
            book.pending.append(o)                         # no open price today: try again tomorrow
            continue
        if o.weight <= 0:
            _close(book, p, p.value, date, o.reason or "signal", c)
        else:
            cut = p.value - o.weight * nav_open
            book.cash += cut * (1 - c)
            p.invested *= (p.value - cut) / p.value
            p.value -= cut
    for o in buys:
        if o.symbol not in opens:
            o_px = md.px("O", i, o.symbol)
            if not np.isfinite(o_px):
                continue                                   # not tradable today: order dropped
            opens[o.symbol] = o_px
        have = book.positions.get(o.symbol)
        if o.weight is not None:
            usd = o.weight * nav_open - (have.value if have else 0.0)
        else:
            usd = (o.cash_frac or 0.0) * book.cash
        usd = min(usd, book.cash)
        if usd <= 1.0:
            continue
        book.cash -= usd
        if have:
            have.value += usd / (1 + c)
            have.invested += usd
        else:
            book.positions[o.symbol] = Position(o.symbol, str(date.date()), usd / (1 + c), usd, tp=o.tp, sl=o.sl,
                                                max_days=o.max_days, last_i=-1, reason=o.reason)
    # ---- 3. intraday exit lines ----
    for p in list(book.positions.values()):
        o, h, l = opens.get(p.symbol), md.px("H", i, p.symbol), md.px("L", i, p.symbol)
        if o is None or not (np.isfinite(h) and np.isfinite(l)) or (p.tp is None and p.sl is None):
            continue
        g_open = p.g
        if p.sl is not None and g_open * l / o <= 1 - p.sl:
            exit_g = min(g_open, 1 - p.sl)
            _close(book, p, p.value * exit_g / g_open, date, "stop line", c)
        elif p.tp is not None and g_open * h / o >= 1 + p.tp:
            exit_g = max(g_open, 1 + p.tp)
            _close(book, p, p.value * exit_g / g_open, date, "profit line", c)
    # ---- 4. close: marks, holding time, missing data ----
    for p in list(book.positions.values()):
        cl = md.px("C", i, p.symbol)
        o = opens.get(p.symbol)
        if np.isfinite(cl) and o is not None and o > 0:
            p.value *= cl / o
            p.g *= cl / o
            p.last_i, p.missing = i, 0
        elif np.isfinite(cl):
            p.last_i, p.missing = i, 0
        else:
            p.missing += 1
        p.days += 1
        if p.missing >= MISSING_DAYS:
            _close(book, p, p.value, date, "no more prices", c)
        elif p.max_days is not None and p.days >= p.max_days:
            _close(book, p, p.value, date, "time", c)
    book.history.append({"date": str(date.date()), "nav": round(book.nav, 2), "cash": round(book.cash, 2),
                         "n_pos": len(book.positions)})
    book.last_date = str(date.date())


def _relink(book: Book, md: MarketData) -> None:
    """After loading a stored book: point each position at the last close we have for it."""
    if book.last_date is None:
        return
    last = md.dates.searchsorted(pd.Timestamp(book.last_date), side="right") - 1
    for p in book.positions.values():
        j = md.col.get(p.symbol)
        p.last_i = -1
        if j is None:
            continue
        col = md._np["C"][: last + 1, j]
        ok = np.flatnonzero(np.isfinite(col))
        p.last_i = int(ok[-1]) if len(ok) else -1


def run(st: Strategy, md: MarketData, book: Book, first: int, last: int | None = None) -> Book:
    """Process days first..last: each day = execute yesterday's orders, then decide after the close."""
    last = len(md.dates) - 1 if last is None else last
    for i in range(first, last + 1):
        if book.last_date is not None and md.dates[i] <= pd.Timestamp(book.last_date):
            continue
        if book.last_date is None:
            book.last_date = str(md.dates[i].date())     # first day: only decide
        else:
            step(st, md, book, i)
        book.pending = list(book.pending) + list(st.decide(i, book, md) or [])
    return book


def backtest(st: Strategy, md: MarketData, start=None, end=None, prepared: bool = False) -> Book:
    """start / end (dates) limit the signal days, e.g. to split history into a training and a test period.
    Positions still open at `end` stay open (valued at the last close)."""
    if not prepared:
        st.prepare(md)
    book = Book(cash=st.capital)
    first = st.first_day(md)
    if start is not None:
        first = max(first, int(md.dates.searchsorted(pd.Timestamp(start))))
    last = None if end is None else int(md.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    run(st, md, book, first, last)
    return book


def live(st: Strategy, md: MarketData, folder: Path) -> Book:
    """Advance the strategy's paper account to the latest day. The first run only places orders."""
    book = Book.load(folder, st.capital)
    _relink(book, md)
    st.prepare(md)
    first = len(md.dates) - 1 if book.last_date is None else 0
    run(st, md, book, first)
    book.save(folder)
    return book


# ============================== Metrics ==============================
def metrics(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return {}
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    out = {"ann_return": ann, "ann_vol": r.std() * np.sqrt(252),
           "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, "max_drawdown": dd}
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        bann = (b.iloc[-1] / b.iloc[0]) ** (252 / len(r)) - 1
        out["spy_ann"] = bann
        out["excess_ann"] = ann - bann
    return out
