#!/usr/bin/env python3
"""
event.py - event-driven trading: buy a stock when its setup looks right, then leave it alone until it hits
the take-profit line, the stop-loss line, or the maximum holding time. No periodic reshuffling.

Setups (SETUPS)
  stealth   quiet accumulation in small/mid caps: volume rising steadily while the price moves sideways.
            Hold for months: +50% take-profit, -20% stop-loss, at most 126 trading days (~6 months).
  swing     hot, high-attention stocks after a burst of price and volume (the closest US analogue of chasing
            limit-ups). Hold for days: +8% take-profit, -5% stop-loss, at most 10 trading days.

How a trade works (identical in the backtest and in the live list)
  - after the close of day t: candidate pool = stocks passing the setup's universe and pre-filter rules
  - a LightGBM model estimates P(the take-profit line is hit before the stop-loss line within the holding
    period) for every candidate. Training labels are built with exactly those exit rules.
  - buy at the open of t+1 if the probability is in the model's top TOP_FRAC and there is a free slot
  - every day afterwards: low <= stop line -> sell at the stop (or at the open if it gaps below);
    high >= profit line -> sell at the line (or at the open if it gaps above); if both happen on the same day
    the stop is assumed first (conservative); after max_days, sell at the close
  - costs: cost_bps on each side (small caps have wider spreads than large caps)

Honest testing
  - walk-forward: the model for year Y is trained only on trades whose outcome was known before Y starts
  - baseline: the same pool, the same exits and the same slots, but random picks (RANDOM_RUNS seeds).
    The model is only useful if it beats random picks clearly; the exit rules alone make an equity curve.
  - delisted stocks are included where Alpaca has their history (a missing price closes the trade at the last
    close, so a final crash before delisting can be missed: results may be somewhat optimistic)

Usage
  python event.py backtest --setup stealth
  python event.py backtest --setup swing
  python event.py backtest --setup stealth --tp 1.0 --sl 0.25 --max-days 189     # try other exit lines
  python event.py today --setup stealth        # today's candidates + checks on your open positions
Positions file for `today` (optional, edit in Excel): portfolio/event_<setup>.csv
  symbol,entry_date,entry_price,shares
Output: reports/event_<setup>_nav.csv, _trades.csv, _yearly.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import evaluate as E

PORTFOLIO_DIR = Path(os.environ.get("PORTFOLIO_DIR", Path(__file__).resolve().parent / "portfolio"))


@dataclass(frozen=True)
class Setup:
    name: str
    desc: str
    min_price: float
    adv_min: float           # 60-day average dollar volume, lower bound
    adv_max: float           # upper bound (excludes the most-followed large caps for "stealth")
    tp: float                # take-profit, e.g. 0.5 = +50%
    sl: float                # stop-loss, e.g. 0.2 = -20%
    max_days: int            # maximum holding period in trading days
    max_pos: int             # concurrent positions
    max_new: int             # new positions per day at most
    cost_bps: float          # cost per side
    capital: float = 100_000.0


SETUPS = {
    "stealth": Setup("stealth", "quiet accumulation in small/mid caps, hold months",
                     min_price=3.0, adv_min=1e6, adv_max=30e6, tp=0.50, sl=0.20, max_days=126,
                     max_pos=10, max_new=1, cost_bps=25),
    "swing": Setup("swing", "hot stocks after a price and volume burst, hold days",
                   min_price=5.0, adv_min=20e6, adv_max=np.inf, tp=0.08, sl=0.05, max_days=10,
                   max_pos=5, max_new=2, cost_bps=10),
}

TOP_FRAC = 0.10            # buy only candidates in the top 10% of predicted probability
MIN_TRAIN_YEARS = 3        # first test year needs this many years of labelled history
SAMPLE_EVERY = 3           # training rows: every 3rd day (overlapping outcomes add little information)
RANDOM_RUNS = 20
LGBM = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1,
            random_state=7)

log = logging.getLogger("event")


# ============================== Data ==============================
def load_panels(setup: Setup, end=None) -> dict:
    """Adjusted OHLC, raw close, volume for every stock that ever passes the setup's price/liquidity filter."""
    t0 = time.time()
    base = D.load_daily(end=end, columns=["close", "volume"])
    base = base[~base["symbol"].isin(D.BENCHMARKS)]
    close = base.pivot(index="date", columns="symbol", values="close").sort_index().astype("float32")
    vol = base.pivot(index="date", columns="symbol", values="volume").reindex_like(close).astype("float32")
    del base
    dv = close * vol
    adv = dv.rolling(60, min_periods=50).mean()
    ok = (close >= setup.min_price) & (adv >= setup.adv_min) & (adv <= setup.adv_max)
    syms = list(ok.columns[ok.any()])
    log.info("%d of %d stocks ever pass the %s price/liquidity filter (%.0f s)",
             len(syms), close.shape[1], setup.name, time.time() - t0)
    df = D.load_daily(end=end, symbols=syms, adjusted=True, columns=["open", "high", "low", "close", "volume", "ret"])
    piv = lambda c: df.pivot(index="date", columns="symbol", values=c).reindex(index=close.index, columns=syms).astype("float32")
    pn = {"O": piv("adj_open"), "H": piv("adj_high"), "L": piv("adj_low"), "C": piv("adj_close"),
          "P": close[syms], "V": vol[syms], "ADV": adv[syms], "UNI": ok[syms]}
    del df
    spy = D.load_benchmark("SPY").set_index("date")["adj_close"].reindex(close.index).ffill()
    pn["SPY"] = spy
    log.info("Panels ready: %d days x %d stocks (%.0f s)", len(close), len(syms), time.time() - t0)
    return pn


def _trend_corr(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """Rolling correlation of each column with time (+1 = steadily rising)."""
    t = pd.Series(np.arange(len(x), dtype="float64"), index=x.index)
    mt, vt = t.rolling(w).mean(), t.rolling(w).var()
    mx = x.rolling(w, min_periods=int(w * 0.8)).mean()
    cov = (x.mul(t, axis=0)).rolling(w, min_periods=int(w * 0.8)).mean() - mx.mul(mt, axis=0)
    sx = x.rolling(w, min_periods=int(w * 0.8)).std(ddof=0)
    return cov.div(np.sqrt(vt * (w - 1) / w), axis=0) / sx


def features(pn: dict) -> dict[str, pd.DataFrame]:
    """Price/volume features known at the close of each day."""
    C, O, H, L, V = pn["C"], pn["O"], pn["H"], pn["L"], pn["V"]
    R = C.pct_change(fill_method=None)
    lv = np.log(V.where(V > 0))
    lc = np.log(C)
    f = {}
    for n in (1, 5, 20, 60, 120, 250):
        f[f"ret_{n}"] = C / C.shift(n) - 1
    f["vol_20"] = R.rolling(20, min_periods=15).std()
    f["vol_60"] = R.rolling(60, min_periods=45).std()
    hi60, lo60 = C.rolling(60, min_periods=45).max(), C.rolling(60, min_periods=45).min()
    f["range_60"] = hi60 / lo60 - 1                              # small = sideways
    f["close_pos_60"] = (C - lo60) / (hi60 - lo60)
    f["dist_high_250"] = C / C.rolling(250, min_periods=200).max()
    f["breakout_20"] = C / C.shift(1).rolling(20, min_periods=15).max()
    f["vol_ratio_5_60"] = np.log(V.rolling(5).mean() / V.rolling(60, min_periods=45).mean())
    f["vol_ratio_20_120"] = np.log(V.rolling(20, min_periods=15).mean() / V.rolling(120, min_periods=90).mean())
    f["vol_trend_60"] = _trend_corr(lv, 60)                     # +1 = volume rising steadily
    f["price_trend_60"] = _trend_corr(lc, 60)
    obv = (np.sign(R).fillna(0) * V).cumsum()
    f["obv_minus_price_60"] = _trend_corr(obv, 60) - f["price_trend_60"]   # accumulation divergence
    up = V.where(R > 0, 0.0).rolling(20, min_periods=15).sum()
    dn = V.where(R < 0, 0.0).rolling(20, min_periods=15).sum()
    f["updown_vol_20"] = np.log((up + 1) / (dn + 1))
    f["gap_20"] = np.log(O / C.shift(1)).rolling(20, min_periods=15).sum()
    f["max_ret_20"] = R.rolling(20, min_periods=15).max()
    f["hl_range_20"] = np.log(H / L).rolling(20, min_periods=15).mean()
    f["log_adv"] = np.log(pn["ADV"])
    f["log_price"] = np.log(pn["P"])
    spy = pn["SPY"]
    bcast = lambda s: pd.DataFrame(np.broadcast_to(s.to_numpy()[:, None], C.shape), index=C.index, columns=C.columns)
    f["spy_trend_200"] = bcast(spy / spy.rolling(200, min_periods=150).mean() - 1)
    f["spy_ret_20"] = bcast(spy / spy.shift(20) - 1)
    return f


def candidate_pool(setup: Setup, pn: dict, f: dict) -> pd.DataFrame:
    """Boolean date x symbol: stocks that are candidates after the close of each day."""
    uni = pn["UNI"] & pn["O"].notna()
    if setup.name == "stealth" or setup.name.startswith("stealth"):
        pre = (f["vol_ratio_20_120"] >= np.log(1.2)) & (f["ret_60"].abs() <= 0.15) & (f["range_60"] <= 0.40) \
            & (f["vol_trend_60"] > 0)
    else:
        pre = (f["ret_5"] >= 0.10) | (f["vol_ratio_5_60"] >= np.log(2.0))
    return (uni & pre).fillna(False)


# ============================== Trade outcomes ==============================
def outcomes(pn: dict, tp: float, sl: float, K: int):
    """For a buy at the open of t+1 (signal after the close of t), the exit return, holding days and
    result (+1 profit line, -1 stop line, 0 time exit / data ended). Gross of costs. Arrays T x N."""
    O, H, L, C = (pn[k].to_numpy(dtype="float32") for k in ("O", "H", "L", "C"))
    T, N = C.shape
    pad = np.full((K + 1, N), np.nan, dtype="float32")
    Op, Hp, Lp, Cp = (np.vstack([x, pad]) for x in (O, H, L, C))
    entry = Op[1:T + 1]
    ret = np.full((T, N), np.nan, dtype="float32")
    days = np.zeros((T, N), dtype="int16")
    res = np.zeros((T, N), dtype="int8")
    done = ~np.isfinite(entry)
    last = np.full((T, N), np.nan, dtype="float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(1, K + 1):
            o, h, l, c = Op[k:k + T] / entry, Hp[k:k + T] / entry, Lp[k:k + T] / entry, Cp[k:k + T] / entry
            have = np.isfinite(c)
            gone = ~done & ~have & np.isfinite(last)                 # price series ended: exit at last close
            ret[gone], days[gone], done[gone] = last[gone] - 1, k - 1, True
            live = ~done & have
            if k == 1:
                o = np.where(np.isfinite(o), 1.0, o)                  # filled at the open by construction
            stop = live & (l <= 1 - sl)
            ret[stop] = np.minimum(o[stop], 1 - sl) - 1
            prof = live & ~stop & (h >= 1 + tp)
            ret[prof] = np.maximum(o[prof], 1 + tp) - 1
            hit = stop | prof
            days[hit], res[stop], res[prof] = k, -1, 1
            done |= hit
            last = np.where(have, c, last)
            if k == K:
                t_out = ~done & np.isfinite(last)
                ret[t_out], days[t_out] = last[t_out] - 1, K
    return ret, days, res


# ============================== Model ==============================
@dataclass
class Dataset:
    ti: np.ndarray           # row date index
    ni: np.ndarray           # row stock index
    X: np.ndarray            # rows x features
    names: list
    ret: np.ndarray          # gross trade return
    days: np.ndarray
    res: np.ndarray


def build_dataset(pool: pd.DataFrame, f: dict, ret, days, res) -> Dataset:
    ti, ni = np.nonzero(pool.to_numpy())
    names = list(f)
    X = np.empty((len(ti), len(names)), dtype="float32")
    for j, n in enumerate(names):
        X[:, j] = f[n].to_numpy(dtype="float32")[ti, ni]
    X[~np.isfinite(X)] = np.nan
    return Dataset(ti, ni, X, names, ret[ti, ni], days[ti, ni], res[ti, ni])


class _Model:
    """Thin wrapper around a native LightGBM booster (no scikit-learn needed)."""
    def __init__(self, booster):
        self.booster = booster

    def prob(self, X):
        return self.booster.predict(X)


def _fit(X, y) -> _Model:
    try:
        import lightgbm as lgb
    except ImportError:
        raise SystemExit("event.py needs LightGBM: pip install lightgbm")
    params = {k: v for k, v in LGBM.items() if k not in ("n_estimators", "random_state", "n_jobs")}
    params.update(seed=LGBM["random_state"], num_threads=0)
    rounds = LGBM["n_estimators"]
    return _Model(lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via ranks (0.5 = no skill)."""
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def walk_forward(ds: Dataset, dates: pd.DatetimeIndex, K: int) -> tuple[np.ndarray, dict]:
    """Out-of-sample probability for every row of the test years, plus each year's buy threshold."""
    prob = np.full(len(ds.ti), np.nan)
    thresholds = {}
    years = sorted(set(dates.year))
    first = dates[0]
    label_ok = np.isfinite(ds.ret)
    for y in years:
        start = dates.searchsorted(pd.Timestamp(year=y, month=1, day=1))
        if start >= len(dates) or (dates[start] - first).days < 365 * MIN_TRAIN_YEARS:
            continue
        end = dates.searchsorted(pd.Timestamp(year=y + 1, month=1, day=1))
        train = label_ok & (ds.ti + K + 1 < start) & (ds.ti % SAMPLE_EVERY == 0)
        test = (ds.ti >= start) & (ds.ti < end)
        if train.sum() < 2000 or not test.any():
            continue
        y_tr = (ds.res[train] == 1).astype(int)
        m = _fit(ds.X[train], y_tr)
        p_tr = m.prob(ds.X[train])
        thresholds[y] = float(np.quantile(p_tr, 1 - TOP_FRAC))
        prob[test] = m.prob(ds.X[test])
        log.info("%d: trained on %d trades (win rate %.0f%%), threshold %.3f, %d candidates to score",
                 y, train.sum(), 100 * y_tr.mean(), thresholds[y], test.sum())
    return prob, thresholds


# ============================== Portfolio simulation ==============================
def simulate(setup: Setup, pn: dict, ds: Dataset, pick: np.ndarray, rank: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pick: rows eligible to buy; rank: higher = preferred. Slots are equal-sized (cash / free slots)."""
    dates = pn["C"].index
    C = pn["C"].to_numpy()
    O = pn["O"].to_numpy()
    syms = pn["C"].columns
    c = setup.cost_bps / 1e4
    order = np.lexsort((-rank[pick], ds.ti[pick]))
    rows = np.flatnonzero(pick)[order]
    by_day: dict[int, list[int]] = {}
    for r in rows:
        by_day.setdefault(int(ds.ti[r]), []).append(int(r))
    cash, pos, recs, trades = setup.capital, [], [], []
    started = False
    lastpx = {}
    for i in range(len(dates)):
        # entries at today's open from yesterday's signals (exits happen later in the day)
        cand = by_day.get(i - 1, [])
        if cand:
            held = {p["j"] for p in pos}
            n_new = 0
            for r in cand:
                if len(pos) >= setup.max_pos or n_new >= setup.max_new:
                    break
                j = int(ds.ni[r])
                if j in held or not np.isfinite(ds.ret[r]) or not np.isfinite(O[i, j]):
                    continue
                budget = cash / (setup.max_pos - len(pos))
                shares = budget / (O[i, j] * (1 + c))
                cash -= budget
                pos.append({"j": j, "t": i - 1, "entry": float(O[i, j]), "shares": shares, "ret": float(ds.ret[r]),
                            "days": int(ds.days[r]), "res": int(ds.res[r]), "exit_i": i - 1 + int(ds.days[r])})
                held.add(j)
                n_new += 1
                started = True
        # exits during today (profit line, stop line or time)
        keep = []
        for p in pos:
            if p["exit_i"] == i:
                value = p["shares"] * p["entry"] * (1 + p["ret"]) * (1 - c)
                cash += value
                trades.append({"symbol": syms[p["j"]], "signal_date": dates[p["t"]], "entry_date": dates[p["t"] + 1],
                               "exit_date": dates[i], "days": p["days"], "result": p["res"],
                               "ret_gross": p["ret"], "ret_net": (1 + p["ret"]) * (1 - c) / (1 + c) - 1})
            else:
                keep.append(p)
        pos = keep
        if not started:
            continue
        mv = 0.0
        for p in pos:
            px = C[i, p["j"]]
            if np.isfinite(px):
                lastpx[p["j"]] = px
            mv += p["shares"] * lastpx.get(p["j"], p["entry"])
        recs.append({"date": dates[i], "nav": cash + mv, "n_pos": len(pos), "cash": cash})
    nav = pd.DataFrame(recs).set_index("date") if recs else pd.DataFrame(columns=["nav", "n_pos", "cash"])
    return nav, pd.DataFrame(trades)


def _ann(nav: pd.Series) -> tuple[float, float, float, float]:
    r = nav.pct_change().dropna()
    if len(r) < 20:
        return (np.nan,) * 4
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return ann, vol, (r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan), dd


# ============================== Backtest ==============================
def backtest(setup: Setup) -> None:
    pn = load_panels(setup)
    t0 = time.time()
    f = features(pn)
    pool = candidate_pool(setup, pn, f)
    ret, days, res = outcomes(pn, setup.tp, setup.sl, setup.max_days)
    ds = build_dataset(pool, f, ret, days, res)
    del f
    dates = pn["C"].index
    log.info("Features, pool and trade outcomes: %d candidate rows (%.0f s)", len(ds.ti), time.time() - t0)
    prob, thr = walk_forward(ds, dates, setup.max_days)
    tested = np.isfinite(prob)
    if not tested.any():
        raise SystemExit("Not enough history for a walk-forward test")
    c2 = 2 * setup.cost_bps / 1e4
    year = dates.year.to_numpy()[ds.ti]
    top = tested & (prob >= np.array([thr.get(y, np.inf) for y in year]))
    known = np.isfinite(ds.ret)

    # ---- trade-level quality: model picks vs every candidate (no slot limits) ----
    rows = []
    for y in sorted(set(year[tested])):
        a = tested & known & (year == y)
        t_ = a & top
        auc = _auc(ds.res[a] == 1, prob[a])
        rows.append({"year": y, "candidates": int(a.sum()), "all_win%": 100 * (ds.res[a] == 1).mean(),
                     "all_avg_net%": 100 * (ds.ret[a].mean() - c2), "picked": int(t_.sum()),
                     "picked_win%": 100 * (ds.res[t_] == 1).mean() if t_.any() else np.nan,
                     "picked_avg_net%": 100 * (ds.ret[t_].mean() - c2) if t_.any() else np.nan, "AUC": auc})
    yt = pd.DataFrame(rows).set_index("year")

    # ---- portfolio: model vs random picks from the same pool ----
    nav, trades = simulate(setup, pn, ds, top, np.nan_to_num(prob))
    rng = np.random.default_rng(0)
    rand_stats = []
    for _ in range(RANDOM_RUNS):
        rr = rng.random(len(prob))
        rpick = tested & (rr >= 1 - TOP_FRAC)
        rnav, _ = simulate(setup, pn, ds, rpick, rr)
        if len(rnav) > 20:
            rand_stats.append(_ann(rnav["nav"]))
    spy = pn["SPY"].reindex(nav.index)
    m_ann = _ann(nav["nav"])
    s_ann = _ann(spy)
    r_ann = np.array([s[0] for s in rand_stats])

    E.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    nav.assign(spy=spy).to_csv(E.REPORT_DIR / f"event_{setup.name}_nav.csv")
    trades.to_csv(E.REPORT_DIR / f"event_{setup.name}_trades.csv", index=False)
    yt.to_csv(E.REPORT_DIR / f"event_{setup.name}_yearly.csv")

    pd.set_option("display.width", 200)
    print(f"\n===== event.py [{setup.name}]: {setup.desc} =====")
    print(f"exits: +{setup.tp:.0%} / -{setup.sl:.0%} / {setup.max_days} days; {setup.max_pos} slots, "
          f"up to {setup.max_new} new per day; cost {setup.cost_bps:g} bps per side; buy top {TOP_FRAC:.0%} by probability")
    print("\n--- Trade quality per year (every candidate vs the model's picks, before slot limits, net of costs) ---")
    print(yt.to_string(float_format=lambda v: f"{v:.2f}"))
    if len(trades):
        w = trades["ret_net"] > 0
        print(f"\n--- Portfolio trades: {len(trades)}, win rate {w.mean():.0%}, avg win {trades.loc[w, 'ret_net'].mean():+.1%}, "
              f"avg loss {trades.loc[~w, 'ret_net'].mean():+.1%}, avg net per trade {trades['ret_net'].mean():+.2%}, "
              f"avg holding {trades['days'].mean():.0f} days")
        print("exit reasons: " + ", ".join(f"{k} {v}" for k, v in
                                          trades["result"].map({1: "profit line", -1: "stop line", 0: "time/data end"})
                                          .value_counts().items()))
    fmt = lambda s: f"ann {s[0]:+.1%}  vol {s[1]:.1%}  Sharpe {s[2]:.2f}  maxDD {s[3]:.1%}"
    print(f"\n--- Portfolio {nav.index[0].date()} to {nav.index[-1].date()} ---")
    print(f"model picks : {fmt(m_ann)}")
    if len(r_ann):
        pct = (r_ann < m_ann[0]).mean()
        print(f"random picks: ann {np.mean(r_ann):+.1%} on average (range {r_ann.min():+.1%} to {r_ann.max():+.1%}); "
              f"the model beats {pct:.0%} of {len(r_ann)} random runs")
    print(f"SPY         : {fmt(s_ann)}")
    yr = nav["nav"].groupby(nav.index.year).last()
    base = yr.shift(1)
    base.iloc[0] = nav["nav"].iloc[0]
    sp = spy.groupby(spy.index.year).last()
    sb = sp.shift(1)
    sb.iloc[0] = spy.iloc[0]
    print("\nCalendar years: " + "  ".join(f"{y}: {v:+.0%} (SPY {s:+.0%})" for y, v, s in
                                            zip(yr.index, yr / base - 1, sp / sb - 1)))
    print(f"\nDetails in {E.REPORT_DIR}/event_{setup.name}_*.csv")


# ============================== Today ==============================
def today(setup: Setup, top_n: int = 15) -> None:
    pn = load_panels(setup)
    f = features(pn)
    pool = candidate_pool(setup, pn, f)
    ret, days, res = outcomes(pn, setup.tp, setup.sl, setup.max_days)
    ds = build_dataset(pool, f, ret, days, res)
    dates, syms = pn["C"].index, pn["C"].columns
    last = len(dates) - 1
    train = np.isfinite(ds.ret) & (ds.ti + setup.max_days + 1 < last) & (ds.ti % SAMPLE_EVERY == 0)
    if train.sum() < 500:
        raise SystemExit(f"Only {train.sum()} past {setup.name} trades to learn from: too few for a model")
    m = _fit(ds.X[train], (ds.res[train] == 1).astype(int))
    thr = float(np.quantile(m.prob(ds.X[train]), 1 - TOP_FRAC))
    now = ds.ti == last
    print(f"\n===== {setup.name} candidates after the close of {dates[last].date()} "
          f"(exits +{setup.tp:.0%} / -{setup.sl:.0%} / {setup.max_days} days) =====")
    if now.any():
        p = m.prob(ds.X[now])
        px = pn["P"].iloc[last].to_numpy()
        t = pd.DataFrame({"symbol": syms[ds.ni[now]], "prob": p, "close": px[ds.ni[now]]})
        t["buy_signal"] = np.where(t["prob"] >= thr, "YES", "")
        t["profit_line"] = t["close"] * (1 + setup.tp)
        t["stop_line"] = t["close"] * (1 - setup.sl)
        print(t.sort_values("prob", ascending=False).head(top_n).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"Buy threshold {thr:.3f} (top {TOP_FRAC:.0%} of training probabilities). "
              f"Lines are from today's close; set them from your actual fill price.")
    else:
        print("No candidates today.")
    pf = PORTFOLIO_DIR / f"event_{setup.name}.csv"
    if pf.exists():
        held = pd.read_csv(pf, parse_dates=["entry_date"])
        print(f"\n--- Your positions ({pf.name}) ---")
        for _, h in held.iterrows():
            s = h["symbol"]
            if s not in pn["P"].columns:
                print(f"{s}: no data")
                continue
            raw = pn["P"][s]
            k = (pn["P"].index > h["entry_date"]).sum()
            r = raw.iloc[-1] / h["entry_price"] - 1
            act = "SELL: profit line reached" if r >= setup.tp else "SELL: stop line reached" if r <= -setup.sl \
                else "SELL: holding period over" if k >= setup.max_days else "hold"
            print(f"{s:<6} entry {h['entry_price']:.2f} on {h['entry_date'].date()}, now {raw.iloc[-1]:.2f} ({r:+.1%}), "
                  f"day {k}/{setup.max_days}, profit line {h['entry_price'] * (1 + setup.tp):.2f}, "
                  f"stop line {h['entry_price'] * (1 - setup.sl):.2f} -> {act}")


def main() -> None:
    p = argparse.ArgumentParser(description="Event-driven setups: buy on a signal, exit on profit / stop / time")
    p.add_argument("cmd", choices=["backtest", "today"])
    p.add_argument("--setup", choices=list(SETUPS), default="stealth")
    p.add_argument("--tp", type=float, help="take-profit, e.g. 0.5 = +50%%")
    p.add_argument("--sl", type=float, help="stop-loss, e.g. 0.2 = -20%%")
    p.add_argument("--max-days", type=int)
    p.add_argument("--slots", type=int, help="concurrent positions")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    s = SETUPS[args.setup]
    over = {k: v for k, v in {"tp": args.tp, "sl": args.sl, "max_days": args.max_days, "max_pos": args.slots}.items()
            if v is not None}
    s = replace(s, **over)
    backtest(s) if args.cmd == "backtest" else today(s)


if __name__ == "__main__":
    main()
