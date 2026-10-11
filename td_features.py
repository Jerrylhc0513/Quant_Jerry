#!/usr/bin/env python3
"""
td_features.py - TechDist features and labels (step P1)

One row per (date, stock) for every CANDIDATE from 2020 on. Everything in a row is known at the
close of `date`; the labels describe what happens after the next open.

Labels (decision after the close of day t, entry at the open of t+1, exit at the close of t+h)
  y_h{h}    log(close[t+h] / open[t+1])                       h in HORIZONS
  ex_h{h}   y_h{h} - beta60 * (same return of QQQ)            market-neutral version, for ranking
  rvf_h{h}  realised variance over the same window             used to fit the volatility model
  end_h{h}  the date the label is complete (t+h); training rows must end before the refit date

Feature groups (see FEATURES): price, volume, volatility (minute based where possible),
cross-asset, regime, events. All ratios / log returns, so they do not depend on the level
the adjusted prices happen to be scaled to.

Usage
  python td_features.py            # build data/td/features.parquet
  python td_features.py --check    # look-ahead test: rebuild on data cut at a few dates and compare
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import td_data as T

log = logging.getLogger("td_features")

FEATURES_FILE = T.TD_DIR / "features.parquet"
HORIZONS = (1, 3, 5)
EPS = 1e-12

FEATURES = [
    # price
    "r1", "r3", "r5", "r10", "r20", "r60",
    "ma5_dist", "ma20_dist", "ma50_dist", "ma200_dist", "zp_vwap20",
    "down_streak", "up_streak", "dd20", "runup10_max20", "close_loc", "gap", "intraday_ret",
    # volume
    "zv", "vol_trend", "updown_vol_5", "last30_vshare_5", "first30_vshare", "log_amihud20",
    # volatility
    "rv_d", "rv_w", "rv_m", "rv_trend", "rskew_5", "jump_share_5", "open30_ret", "last30_ret",
    # cross-asset
    "beta60", "resid_r1", "resid_r5", "resid_r20", "rank_r5", "rank_resid_r5",
    # regime
    "qqq_r1", "qqq_r5", "qqq_r20", "qqq_ma200_dist", "qqq_rv_m", "smh_r5", "smh_r20", "tlt_r20",
    "avg_corr60",
    # events and calendar
    "days_since_earn", "days_to_earn_est", "earn_soon", "fomc_next", "cpi_next", "dow", "days_to_month_end",
    # stock identity (categorical)
    "sym_id",
]


def _lr(a, b):
    return np.log(a / b)


def _streak(sign: pd.DataFrame) -> pd.DataFrame:
    """Number of consecutive days (up to today) on which `sign` is True."""
    s = sign.fillna(False).astype(int)
    return s.apply(lambda col: col.groupby((col == 0).cumsum()).cumsum())


def _event_features(idx: pd.DatetimeIndex, stocks: list[str], events: pd.DataFrame):
    """days since the last known earnings day, and days to the next one estimated as last + 63."""
    since = pd.DataFrame(np.nan, index=idx, columns=stocks)
    to = pd.DataFrame(np.nan, index=idx, columns=stocks)
    n = len(idx)
    for s in stocks:
        ev = events[events["symbol"] == s]
        if ev.empty:
            continue
        d_pos = idx.searchsorted(ev["date"].values)
        k_pos = idx.searchsorted(ev["known"].values)
        last = np.full(n, np.nan)
        for dp, kp in sorted(zip(d_pos, k_pos), key=lambda x: x[1]):
            if kp >= n:
                continue
            seg = last[kp:]
            last[kp:] = np.where(np.isnan(seg) | (seg < dp), dp, seg)
        t = np.arange(n)
        since[s] = t - last
        to[s] = np.clip(last + 63 - t, -10, 63)
    return since, to


def build(P: dict, intraday: pd.DataFrame, events: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    C, O, H, L, V, VW, Craw = P["C"], P["O"], P["H"], P["L"], P["V"], P["VWAP"], P["close_raw"]
    idx = C.index
    stocks = [s for s in T.CANDIDATES if s in C.columns]
    q = "QQQ"
    lc = np.log(C)
    r = lc.diff()
    sig20 = r.rolling(20, min_periods=15).std()
    F: dict[str, pd.DataFrame] = {}

    # ---- price ----
    for k in (1, 3, 5, 10, 20, 60):
        F[f"r{k}"] = lc - lc.shift(k)
    for k in (5, 20, 50, 200):
        F[f"ma{k}_dist"] = _lr(C, C.rolling(k, min_periods=int(k * 0.8)).mean()) / sig20
    vw20 = (VW * V).rolling(20, min_periods=15).sum() / V.rolling(20, min_periods=15).sum()
    F["zp_vwap20"] = _lr(C, vw20) / sig20
    F["down_streak"] = _streak(r < 0)
    F["up_streak"] = _streak(r > 0)
    F["dd20"] = _lr(C, C.rolling(20, min_periods=15).max())
    r10 = lc - lc.shift(10)
    F["runup10_max20"] = r10.rolling(20, min_periods=10).max()
    F["close_loc"] = ((C - L) / (H - L)).where(H > L, 0.5)
    F["gap"] = _lr(O, C.shift(1))
    F["intraday_ret"] = _lr(C, O)

    # ---- volume ----
    F["zv"] = np.log((V + 1) / (V.rolling(20, min_periods=15).mean().shift(1) + 1))
    F["vol_trend"] = np.log((V.rolling(5).mean() + 1) / (V.rolling(20, min_periods=15).mean() + 1))
    dv = Craw * V
    F["log_amihud20"] = np.log((r.abs() / dv.where(dv > 0)).rolling(20, min_periods=15).mean() * 1e9 + EPS)

    # ---- intraday (minute) summaries ----
    def wide(col):
        if intraday.empty or col not in intraday:
            return pd.DataFrame(np.nan, index=idx, columns=C.columns)
        w = intraday.pivot(index="date", columns="symbol", values=col)
        return w.reindex(index=idx, columns=C.columns)

    rv_intra = wide("rv_intra")
    gk = 0.5 * _lr(H, L) ** 2 - (2 * np.log(2) - 1) * _lr(C, O) ** 2
    intra_var = (rv_intra ** 2).where(rv_intra.notna(), gk)
    rvday = intra_var + F["gap"].fillna(0) ** 2
    F["rv_d"] = np.log(rvday + EPS)
    F["rv_w"] = np.log(rvday.rolling(5, min_periods=4).mean() + EPS)
    F["rv_m"] = np.log(rvday.rolling(22, min_periods=15).mean() + EPS)
    F["rv_trend"] = F["rv_w"] - F["rv_m"]
    F["rskew_5"] = wide("rskew").rolling(5, min_periods=3).mean()
    F["jump_share_5"] = wide("jump_share").rolling(5, min_periods=3).mean()
    F["open30_ret"] = wide("open30_ret")
    F["last30_ret"] = wide("last30_ret")
    F["updown_vol_5"] = wide("updown_vol").rolling(5, min_periods=3).mean()
    F["last30_vshare_5"] = wide("last30_vshare").rolling(5, min_periods=3).mean()
    F["first30_vshare"] = wide("first30_vshare")

    # ---- cross-asset ----
    rq = r[q]
    beta = r.rolling(60, min_periods=40).cov(rq).div(rq.rolling(60, min_periods=40).var(), axis=0)
    F["beta60"] = beta
    for k in (1, 5, 20):
        F[f"resid_r{k}"] = F[f"r{k}"] - beta.mul(F[f"r{k}"][q], axis=0)
    F["rank_r5"] = F["r5"][stocks].rank(axis=1, pct=True).reindex(columns=C.columns)
    F["rank_resid_r5"] = F["resid_r5"][stocks].rank(axis=1, pct=True).reindex(columns=C.columns)

    # ---- regime (same value for every stock on a date) ----
    def bcast(series):
        return pd.DataFrame(np.repeat(series.to_numpy()[:, None], len(C.columns), axis=1),
                            index=idx, columns=C.columns)

    for k in (1, 5, 20):
        F[f"qqq_r{k}"] = bcast(F[f"r{k}"][q])
    F["qqq_ma200_dist"] = bcast(_lr(C[q], C[q].rolling(200, min_periods=160).mean()))
    F["qqq_rv_m"] = bcast(F["rv_m"][q])
    F["smh_r5"] = bcast(F["r5"]["SMH"]) if "SMH" in C else bcast(pd.Series(np.nan, index=idx))
    F["smh_r20"] = bcast(F["r20"]["SMH"]) if "SMH" in C else bcast(pd.Series(np.nan, index=idx))
    F["tlt_r20"] = bcast(F["r20"]["TLT"]) if "TLT" in C else bcast(pd.Series(np.nan, index=idx))
    basket = r[stocks].mean(axis=1)
    corr = r[stocks].rolling(60, min_periods=40).corr(basket)
    F["avg_corr60"] = bcast(corr.mean(axis=1))

    # ---- events and calendar ----
    since, to = _event_features(idx, stocks, events)
    F["days_since_earn"] = since.reindex(columns=C.columns)
    F["days_to_earn_est"] = to.reindex(columns=C.columns)
    F["earn_soon"] = ((to >= 0) & (to <= 2)).astype(float).where(to.notna()).reindex(columns=C.columns)
    nxt = pd.Series(idx + pd.offsets.BDay(1), index=idx)      # calendar based: no peeking at the next row
    for name, ev in (("fomc_next", "FOMC"), ("cpi_next", "CPI")):
        days = set(macro.loc[macro["event"] == ev, "date"]) if len(macro) else set()
        F[name] = bcast(nxt.isin(days).astype(float))
    F["dow"] = bcast(pd.Series(idx.dayofweek, index=idx).astype(float))
    month_end = (idx + pd.offsets.MonthEnd(0)).values.astype("datetime64[D]")
    left = np.busday_count(idx.values.astype("datetime64[D]") + np.timedelta64(1, "D"), month_end + np.timedelta64(1, "D"))
    F["days_to_month_end"] = bcast(pd.Series(left.astype(float), index=idx))
    F["sym_id"] = pd.DataFrame({s: float(T.CANDIDATES.index(s)) if s in T.CANDIDATES else np.nan
                                for s in C.columns}, index=idx)

    # ---- labels ----
    LBL: dict[str, pd.DataFrame] = {}
    for h in HORIZONS:
        y = lc.shift(-h) - np.log(O).shift(-1)
        LBL[f"y_h{h}"] = y
        LBL[f"ex_h{h}"] = y - beta.mul(y[q], axis=0)
        rvf = intra_var.shift(-1)
        for j in range(2, h + 1):
            rvf = rvf + rvday.shift(-j)
        LBL[f"rvf_h{h}"] = rvf
        end = pd.Series(idx, index=idx).shift(-h)
        LBL[f"end_h{h}"] = pd.DataFrame(np.repeat(end.to_numpy()[:, None], len(C.columns), axis=1),
                                        index=idx, columns=C.columns)

    # ---- long table ----
    keep = idx >= pd.Timestamp(T.SAMPLE_START)
    parts = {}
    for name in FEATURES:
        parts[name] = F[name].loc[keep, stocks].stack(future_stack=True)
    for name, w in LBL.items():
        parts[name] = w.loc[keep, stocks].stack(future_stack=True)
    out = pd.DataFrame(parts)
    out.index.names = ["date", "symbol"]
    out = out.reset_index()
    num = out.select_dtypes("number").columns
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    return out[out["r1"].notna()].reset_index(drop=True)


def load_inputs(cut=None):
    """Daily panels, intraday summary, events and macro calendar, optionally cut at a date."""
    P = T.daily_panels(T.CANDIDATES + T.ETFS)
    intraday = T.load_intraday()
    if cut is not None:
        cut = pd.Timestamp(cut)
        df = T.load_daily(T.CANDIDATES + T.ETFS)
        df = df[df["date"] <= cut].drop(columns=[c for c in df if c.startswith("adj_")])
        import data as D
        df = D.add_adjusted(df)
        df["adj_vwap"] = df["vwap"] * df["adj_close"] / df["close"]
        P = {}
        for name, col in [("O", "adj_open"), ("H", "adj_high"), ("L", "adj_low"), ("C", "adj_close"),
                          ("close_raw", "close"), ("V", "volume"), ("VWAP", "adj_vwap"), ("ret", "ret")]:
            P[name] = df.pivot(index="date", columns="symbol", values=col).sort_index()
        intraday = intraday[pd.to_datetime(intraday["date"]) <= cut]
    events = T.build_events({k: v[[c for c in v.columns if c in T.CANDIDATES]] for k, v in P.items()},
                            save=False)
    return P, intraday, events, T.load_macro()


def check(cuts=None, tol=1e-6) -> bool:
    """Rebuild features with all data after `cut` removed; every feature on dates <= cut must match."""
    full = build(*load_inputs()).set_index(["date", "symbol"])
    dates = full.index.get_level_values("date").unique().sort_values()
    cuts = cuts or [dates[len(dates) // 4], dates[len(dates) // 2], dates[-30]]
    ok = True
    for cut in cuts:
        part = build(*load_inputs(cut)).set_index(["date", "symbol"])
        a = full.loc[part.index, FEATURES]
        b = part[FEATURES]
        bad = []
        for c in FEATURES:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            same = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tol * (1 + np.abs(x)))
            if not same.all():
                bad.append(f"{c} ({(~same).sum()} rows)")
        print(f"cut {pd.Timestamp(cut).date()}: " + ("OK" if not bad else "LOOK-AHEAD in " + ", ".join(bad)))
        ok &= not bad
    return ok


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_FILE)


def main() -> None:
    p = argparse.ArgumentParser(description="TechDist features")
    p.add_argument("--check", action="store_true", help="look-ahead test")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.check:
        raise SystemExit(0 if check() else 1)
    df = build(*load_inputs())
    df.to_parquet(FEATURES_FILE, index=False)
    cov = df[FEATURES].notna().mean().sort_values()
    log.info("Features: %d rows, %d stocks, %s to %s", len(df), df["symbol"].nunique(),
             df["date"].min().date(), df["date"].max().date())
    log.info("Lowest coverage: %s", ", ".join(f"{k} {v:.0%}" for k, v in cov.head(6).items()))


if __name__ == "__main__":
    main()
