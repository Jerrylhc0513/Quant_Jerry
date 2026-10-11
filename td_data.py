#!/usr/bin/env python3
"""
td_data.py - data for the TechDist research (a small basket of large tech / AI stocks)

TechDist predicts the distribution of each stock's return over the next few days and trades the
best few names of a ~10-stock basket. Design: docs/TECHDIST.md. This file is step P0 (data).

What it maintains (all under data/td/, independent of the 1000-stock universe files)
  daily.parquet       daily bars 2019-01 onwards for the candidates and the ETFs, raw prices plus
                      total return `ret` (re-downloaded in full on every update: a few requests)
  minute/YYYY-MM.parquet
                      1-minute bars, regular session only, raw prices, for the candidates and QQQ
                      (2019-11 onwards; first download about 2000 requests / 15 minutes, resumable)
  intraday.parquet    one row per stock per day summarising its minute bars (realised vol,
                      skew, jumps, first / last 30 minutes, up-volume vs down-volume, day VWAP)
  pools.parquet       which stocks are "in the basket" on each date (pool A fixed, pool B yearly)
  events.parquet      earnings dates per stock, each with the first date it was known
Optional files you can add yourself
  earnings.csv        symbol,date   real earnings reaction days (first session after the release).
                      If present it replaces the automatic detection for the symbols it lists.
  macro_events.csv    date,event    e.g. 2024-03-20,FOMC  /  2024-03-12,CPI.  Without it the
                      macro-event features are simply zero.

Basket definitions
  Pool A  (your choice, fixed):      POOL_A below. Easy to read but chosen with hindsight.
  Pool B  (point in time, honest):   on the first trading day of each year, the 10 CANDIDATES with
                                     the largest 60-day average dollar volume before that day.
  Compare strategies on both. If A is much better than B, the edge is mostly "picked the winners".
  Models are trained on all CANDIDATES (more data); only the evaluation is restricted to a pool.

Usage
  python td_data.py update       # daily + minute + intraday summary + pools + events
  python td_data.py status
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import time as dtime

import numpy as np
import pandas as pd

import data as D

log = logging.getLogger("td_data")

TD_DIR = D.DATA_DIR / "td"
DAILY_FILE = TD_DIR / "daily.parquet"
MINUTE_DIR = TD_DIR / "minute"
MINUTE_MANIFEST = MINUTE_DIR / "complete.json"
INTRADAY_FILE = TD_DIR / "intraday.parquet"
POOLS_FILE = TD_DIR / "pools.parquet"
EVENTS_FILE = TD_DIR / "events.parquet"
EARNINGS_CSV = TD_DIR / "earnings.csv"
MACRO_CSV = TD_DIR / "macro_events.csv"

DAILY_START = "2019-01-01"      # 200-day averages need ~10 months of warm-up before 2020
MINUTE_START = "2019-11-01"     # intraday features need ~1-2 months of warm-up
SAMPLE_START = "2020-01-01"     # no statistics or training on anything earlier

POOL_A = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "ORCL"]
# Large, heavily traded US tech / internet names that already existed (and were big) in 2019.
# Pool B picks from this list each January, so later stars only enter once they are traded heavily.
CANDIDATES = POOL_A + ["INTC", "CSCO", "ADBE", "CRM", "QCOM", "TXN", "NFLX", "IBM", "INTU", "AMAT",
                       "MU", "PYPL", "ACN", "NOW", "LRCX", "ADI"]
POOL_SIZE = 10
POOL_ADV_DAYS = 60
ETFS = ["QQQ", "SPY", "SMH", "TLT", "XLK"]   # market, semiconductors, rates
MINUTE_SYMBOLS = CANDIDATES + ["QQQ"]
MINUTE_BATCH = 10                            # symbols per request (10k bars per page are shared)

EARN_WINDOW_BEFORE = 10          # reaction day is searched in [filed - 10, filed + 1] trading days
EARN_MIN_GAP = 40                # automatic detection: at least 40 sessions between two events


# ============================== Daily ==============================
def update_daily() -> None:
    end = D.last_complete_date()
    df = D._daily_bars(CANDIDATES + ETFS, pd.Timestamp(DAILY_START), end)
    if df.empty:
        raise SystemExit("No daily bars returned (check the Alpaca keys in .env)")
    TD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DAILY_FILE, index=False)
    log.info("Daily bars: %d symbols, %s to %s", df["symbol"].nunique(), df["date"].min().date(),
             df["date"].max().date())


def load_daily(symbols=None) -> pd.DataFrame:
    """Long daily table with adjusted prices (adj_open / adj_high / adj_low / adj_close)."""
    df = pd.read_parquet(DAILY_FILE)
    if symbols is not None:
        df = df[df["symbol"].isin(set(symbols))]
    return D.add_adjusted(df)


def daily_panels(symbols=None) -> dict[str, pd.DataFrame]:
    """Wide date x symbol panels: O H L C (adjusted), close_raw, volume, vwap (adjusted), ret."""
    df = load_daily(symbols)
    df["adj_vwap"] = df["vwap"] * df["adj_close"] / df["close"]
    out = {}
    for name, col in [("O", "adj_open"), ("H", "adj_high"), ("L", "adj_low"), ("C", "adj_close"),
                      ("close_raw", "close"), ("V", "volume"), ("VWAP", "adj_vwap"), ("ret", "ret")]:
        out[name] = df.pivot(index="date", columns="symbol", values=col).sort_index()
    return out


# ============================== Minute ==============================
def _months(start: str, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def _month_file(m: pd.Timestamp):
    return MINUTE_DIR / f"{m:%Y-%m}.parquet"


def _fetch_month(symbols: list[str], m: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    last = min(m + pd.offsets.MonthEnd(0), end)
    t0, t1 = D._et(m, 9, 30), D._et(last, 16, 0)
    frames = [D.fetch_bars(b, "1Min", t0, t1, "raw") for b in D._chunks(symbols, MINUTE_BATCH)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return D._empty(D.BAR_COLS)
    df = pd.concat(frames, ignore_index=True)
    t = df["ts"].dt.time
    return df[(t >= dtime(9, 30)) & (t < dtime(16, 0))]


def _minute_manifest() -> dict:
    try:
        return json.loads(MINUTE_MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def update_minute() -> None:
    """Download every month that is not complete yet, plus missing symbols in complete months.
    A month counts as complete once it was fetched after its last calendar day."""
    MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    end = D.last_complete_date()
    done = _minute_manifest()
    todo = []
    for m in _months(MINUTE_START, end):
        key = f"{m:%Y-%m}"
        if key not in done or not _month_file(m).exists():
            todo.append((m, MINUTE_SYMBOLS, False))
            continue
        missing = [s for s in MINUTE_SYMBOLS if s not in set(done[key])]
        if missing:
            todo.append((m, missing, True))
    if not todo:
        log.info("Minute bars are up to date")
        return
    n_req = sum(math.ceil(len(s) / MINUTE_BATCH) * 25 for _, s, _ in todo)
    log.info("%d month files to fetch, roughly %d requests / %.0f minutes (safe to interrupt)",
             len(todo), n_req, n_req / D.ALPACA_REQ_PER_MIN)
    t0 = time.time()
    for i, (m, syms, merge) in enumerate(todo, 1):
        key = f"{m:%Y-%m}"
        df = _fetch_month(syms, m, end)
        if merge:
            old = pd.read_parquet(_month_file(m))
            df = pd.concat([old[~old["symbol"].isin(syms)], df], ignore_index=True)
        if len(df):
            df.sort_values(["symbol", "ts"]).to_parquet(_month_file(m), index=False)
        if m + pd.offsets.MonthEnd(0) <= end:                 # month is over: mark it complete
            done[key] = sorted(set(done.get(key, [])) | set(syms))
            MINUTE_MANIFEST.write_text(json.dumps(done))
        log.info("Minute bars %s: %d rows (%d/%d, %.0f s)", key, len(df), i, len(todo), time.time() - t0)


def load_minute(start, end=None, symbols=None) -> pd.DataFrame:
    """1-minute bars (raw prices, ET timestamps) for [start, end]."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    files = [_month_file(m) for m in _months(str(start.date()), end) if _month_file(m).exists()]
    if not files:
        return D._empty(D.BAR_COLS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df["ts"] >= start) & (df["ts"] < end + pd.Timedelta(days=1))]
    if symbols is not None:
        df = df[df["symbol"].isin(set(symbols))]
    return df.reset_index(drop=True)


# ============================== Intraday summary ==============================
def intraday_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Minute bars -> one row per (date, symbol). Only uses bars of that same day."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["date"] = df["ts"].dt.normalize()
    df["m"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute - 570          # minutes since 9:30
    key = [df["symbol"], df["date"]]
    lr = np.log(df["close"]).groupby(key).diff()
    df["lr"] = lr
    df["alr"] = lr.abs()
    df["alr_prev"] = df["alr"].groupby(key).shift(1)
    df["lr2"] = lr ** 2
    df["lr3"] = lr ** 3
    df["bvt"] = df["alr"] * df["alr_prev"]
    df["pv"] = df["vwap"].fillna(df["close"]) * df["volume"]
    df["vol_up"] = df["volume"].where(lr > 0, 0.0)
    df["vol_dn"] = df["volume"].where(lr < 0, 0.0)
    df["vol_first30"] = df["volume"].where(df["m"] < 30, 0.0)
    df["vol_last30"] = df["volume"].where(df["m"] >= 360, 0.0)
    g = df.groupby(["date", "symbol"], sort=True)
    out = pd.DataFrame({
        "rv2": g["lr2"].sum(),
        "r3": g["lr3"].sum(),
        "bv": g["bvt"].sum() * np.pi / 2,
        "n_bars": g.size(),
        "first_open": g["open"].first(),
        "last_close": g["close"].last(),
        "volume": g["volume"].sum(),
        "pv": g["pv"].sum(),
        "vol_up": g["vol_up"].sum(),
        "vol_dn": g["vol_dn"].sum(),
        "vol_first30": g["vol_first30"].sum(),
        "vol_last30": g["vol_last30"].sum(),
        "max_m": g["m"].max(),
    })
    c_1000 = df[df["m"] < 30].groupby(["date", "symbol"])["close"].last()
    c_1530 = df[df["m"] < 360].groupby(["date", "symbol"])["close"].last()
    vt = out["volume"].where(out["volume"] > 0)
    res = pd.DataFrame(index=out.index)
    res["rv_intra"] = np.sqrt(out["rv2"])
    res["rskew"] = (np.sqrt(out["n_bars"]) * out["r3"] / out["rv2"] ** 1.5).where(out["rv2"] > 0)
    res["jump_share"] = (1 - out["bv"] / out["rv2"]).clip(lower=0).where(out["rv2"] > 0)
    res["open30_ret"] = np.log(c_1000.reindex(out.index) / out["first_open"])
    res["last30_ret"] = np.log(out["last_close"] / c_1530.reindex(out.index))
    res["first30_vshare"] = out["vol_first30"] / vt
    res["last30_vshare"] = out["vol_last30"] / vt
    res["updown_vol"] = np.log((out["vol_up"] + 1) / (out["vol_dn"] + 1))
    res["vwap_day_raw"] = out["pv"] / vt
    res["half_day"] = out["max_m"] < 330                                  # closed by 15:00
    half = res["half_day"]
    res.loc[half, ["last30_ret", "last30_vshare"]] = np.nan
    return res.reset_index()


def update_intraday() -> None:
    """Recompute the summary for the last two stored months, keep earlier months as they are."""
    files = sorted(MINUTE_DIR.glob("*.parquet"))
    if not files:
        log.info("No minute bars: intraday summary skipped")
        return
    old = pd.read_parquet(INTRADAY_FILE) if INTRADAY_FILE.exists() else pd.DataFrame(columns=["date"])
    done_months = set(pd.to_datetime(old["date"]).dt.strftime("%Y-%m")) if len(old) else set()
    redo = [f for i, f in enumerate(files) if f.stem not in done_months or i >= len(files) - 2]
    parts = []
    for f in redo:
        x = intraday_summary(pd.read_parquet(f))
        if len(x):
            parts.append(x)
    if len(old):
        old = old[~pd.to_datetime(old["date"]).dt.strftime("%Y-%m").isin({f.stem for f in redo})]
        parts.insert(0, old)
    out = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    out.to_parquet(INTRADAY_FILE, index=False)
    log.info("Intraday summary: %d rows, %d months recomputed", len(out), len(redo))


def load_intraday() -> pd.DataFrame:
    if not INTRADAY_FILE.exists():
        return pd.DataFrame(columns=["date", "symbol"])
    return pd.read_parquet(INTRADAY_FILE)


# ============================== Pools ==============================
def build_pools(P: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """(date, symbol, pool) for pool A and pool B, from SAMPLE_START onwards."""
    P = P or daily_panels(CANDIDATES)
    dates = P["C"].index[P["C"].index >= pd.Timestamp(SAMPLE_START)]
    dv = (P["close_raw"] * P["V"]).reindex(columns=CANDIDATES)
    adv = dv.rolling(POOL_ADV_DAYS, min_periods=POOL_ADV_DAYS // 2).mean().shift(1)   # before the day
    rows = []
    for y in sorted(set(dates.year)):
        yd = dates[dates.year == y]
        first = yd[0]
        top = adv.loc[first].dropna().sort_values(ascending=False).index[:POOL_SIZE]
        rows += [(d, s, "B") for d in yd for s in top]
    rows += [(d, s, "A") for d in dates for s in POOL_A if s in P["C"].columns]
    pools = pd.DataFrame(rows, columns=["date", "symbol", "pool"])
    TD_DIR.mkdir(parents=True, exist_ok=True)
    pools.to_parquet(POOLS_FILE, index=False)
    return pools


def load_pools() -> pd.DataFrame:
    return pd.read_parquet(POOLS_FILE)


# ============================== Events ==============================
def _filing_dates() -> pd.DataFrame:
    """(symbol, filed) for 10-Q / 10-K style quarterly filings, from the SEC fundamentals if present."""
    try:
        import fundamentals as FU
        facts, comp = FU.load_facts(), FU.load_companies()
    except Exception:                                    # noqa: BLE001 - fundamentals are optional
        return pd.DataFrame(columns=["symbol", "filed"])
    if facts.empty or comp.empty:
        return pd.DataFrame(columns=["symbol", "filed"])
    f = facts[facts["item"].isin(["ni_q", "rev_q"])].merge(comp[["symbol", "cik"]], on="cik")
    f = f[f["symbol"].isin(CANDIDATES)]
    return f[["symbol", "filed"]].drop_duplicates().sort_values(["symbol", "filed"]).reset_index(drop=True)


def build_events(P: dict[str, pd.DataFrame] | None = None, save: bool = True) -> pd.DataFrame:
    """Earnings reaction days with the first date each one is known (`known`).

    Priority per symbol: earnings.csv (known on the day itself) > SEC filing dates (the reaction day
    is the highest-volume session in [filed - 10, filed + 1]; known from filed + 1) > pure volume
    detection (volume >= 2.5x its 60-day median and an overnight gap >= 2 daily sigmas, at least
    40 sessions after the previous one; known on the day itself)."""
    P = P or daily_panels(CANDIDATES)
    C, O, V = P["C"], P["O"], P["V"]
    idx = C.index
    vr = V / V.rolling(60, min_periods=20).median().shift(1)
    gap = np.log(O / C.shift(1))
    sig = np.log(C / C.shift(1)).rolling(20, min_periods=10).std().shift(1)
    manual = pd.read_csv(EARNINGS_CSV, parse_dates=["date"]) if EARNINGS_CSV.exists() else \
        pd.DataFrame(columns=["symbol", "date"])
    filings = _filing_dates()
    rows = []
    for s in CANDIDATES:
        if s not in C.columns:
            continue
        m = manual[manual["symbol"] == s]
        if len(m):
            for d in m["date"]:
                pos = idx.searchsorted(d)
                if pos < len(idx):
                    rows.append((s, idx[pos], idx[pos], "manual"))
            continue
        fs = filings[filings["symbol"] == s]["filed"]
        if len(fs):
            for fd in fs:
                p = idx.searchsorted(fd)
                lo, hi = max(0, p - EARN_WINDOW_BEFORE), p + 1
                if hi >= len(idx):                     # not known yet on the last stored date
                    continue
                w = vr[s].iloc[lo:hi + 1]
                if w.notna().any():
                    rows.append((s, w.idxmax(), idx[hi], "sec"))
            continue
        last = -10 ** 6
        hit = (vr[s] >= 2.5) & (gap[s].abs() >= 2 * sig[s])
        for p in np.nonzero(hit.to_numpy())[0]:
            if p - last >= EARN_MIN_GAP:
                rows.append((s, idx[p], idx[p], "volume"))
                last = p
    ev = pd.DataFrame(rows, columns=["symbol", "date", "known", "source"])
    ev = ev.sort_values(["symbol", "date", "known"]).drop_duplicates(["symbol", "date"]).reset_index(drop=True)
    if save:
        TD_DIR.mkdir(parents=True, exist_ok=True)
        ev.to_parquet(EVENTS_FILE, index=False)
    return ev


def load_events() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        return pd.DataFrame(columns=["symbol", "date", "known", "source"])
    return pd.read_parquet(EVENTS_FILE)


def load_macro() -> pd.DataFrame:
    if not MACRO_CSV.exists():
        return pd.DataFrame(columns=["date", "event"])
    m = pd.read_csv(MACRO_CSV, parse_dates=["date"])
    m["event"] = m["event"].str.upper().str.strip()
    return m


# ============================== CLI ==============================
def update() -> None:
    update_daily()
    update_minute()
    update_intraday()
    P = daily_panels(CANDIDATES)
    pools = build_pools(P)
    ev = build_events(P)
    log.info("Pools: %d rows; events: %d (%s)", len(pools), len(ev), ev["source"].value_counts().to_dict())


def status() -> None:
    if not DAILY_FILE.exists():
        print("No TechDist data yet: run python td_data.py update")
        return
    d = pd.read_parquet(DAILY_FILE, columns=["date", "symbol"])
    print(f"Daily: {d['symbol'].nunique()} symbols, {d['date'].min().date()} to {d['date'].max().date()}")
    files = sorted(MINUTE_DIR.glob("*.parquet"))
    print(f"Minute: {len(files)} month files" + (f", {files[0].stem} to {files[-1].stem}" if files else ""))
    it = load_intraday()
    if len(it):
        print(f"Intraday summary: {len(it)} rows, last {pd.Timestamp(it['date'].max()).date()}")
    if POOLS_FILE.exists():
        pools = load_pools()
        b = pools[pools["pool"] == "B"]
        for y, g in b.groupby(b["date"].dt.year):
            print(f"Pool B {y}: {', '.join(sorted(g['symbol'].unique()))}")
    ev = load_events()
    if len(ev):
        print(f"Earnings events: {len(ev)} ({ev['source'].value_counts().to_dict()})")
    print(f"macro_events.csv: {'found' if MACRO_CSV.exists() else 'not found (macro flags = 0)'}")


def main() -> None:
    p = argparse.ArgumentParser(description="TechDist data")
    p.add_argument("cmd", choices=["update", "status"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    update() if args.cmd == "update" else status()


if __name__ == "__main__":
    main()
