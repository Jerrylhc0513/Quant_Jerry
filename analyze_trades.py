#!/usr/bin/env python3
"""
analyze_trades.py - look at every trade of a strategy backtest and find out where the losses come from

  python analyze_trades.py second_wave
  python analyze_trades.py second_wave_novol

Reads reports/strategies/<name>_trades.csv (written by run_strategies.py backtest) and adds, per trade:
  entry_gap      open of the entry day vs the signal-day close (buying into a gap up or down)
  mfe / mae      best / worst point while held (from daily highs and lows), vs the entry price
  day1           return of the first day (entry open -> first close)
  fwd_5..fwd_120 return if the stock had simply been held 5 / 20 / 60 / 120 days, no exits
  spy_fwd_*      SPY over the same windows
  spy_trend      SPY above its 200-day average at entry? (market regime)
  price, adv     share price and 60-day dollar volume at the signal
Then prints: results by exit type, how the losses happen, "were the stocks good but the exits bad?",
results by year / regime / price / liquidity / entry gap, and the 10 best and worst trades.
Output: reports/strategies/<name>_trades_detail.csv
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import data as D
import run_strategies as RS

HORIZONS = (5, 20, 60, 120)
log = logging.getLogger("analyze")


def enrich(trades: pd.DataFrame) -> pd.DataFrame:
    syms = sorted(trades["symbol"].unique())
    daily = D.load_daily(symbols=syms, adjusted=True, columns=["open", "high", "low", "close", "volume"])
    piv = lambda c: daily.pivot(index="date", columns="symbol", values=c).sort_index()
    O, H, L, C = piv("adj_open"), piv("adj_high"), piv("adj_low"), piv("adj_close")
    P, V = piv("close"), piv("volume")
    adv = (P * V).rolling(60, min_periods=20).mean()
    spy = D.load_benchmark("SPY").set_index("date")["adj_close"].reindex(C.index).ffill()
    spy_trend = spy / spy.rolling(200, min_periods=150).mean() - 1
    dates = C.index
    rows = []
    for _, t in trades.iterrows():
        s = t["symbol"]
        e = dates.searchsorted(pd.Timestamp(t["entry_date"]))
        x = dates.searchsorted(pd.Timestamp(t["exit_date"]))
        if e >= len(dates) or s not in C:
            continue
        entry = O[s].iloc[e]
        sig = e - 1
        r = {"entry_px": entry,
             "entry_gap": entry / C[s].iloc[sig] - 1 if sig >= 0 else np.nan,
             "day1": C[s].iloc[e] / entry - 1,
             "mfe": H[s].iloc[e:x + 1].max() / entry - 1,
             "mae": L[s].iloc[e:x + 1].min() / entry - 1,
             "price": P[s].iloc[sig] if sig >= 0 else np.nan,
             "adv_musd": adv[s].iloc[sig] / 1e6 if sig >= 0 else np.nan,
             "spy_trend": spy_trend.iloc[sig] if sig >= 0 else np.nan}
        for h in HORIZONS:
            k = e + h - 1
            r[f"fwd_{h}"] = C[s].iloc[k] / entry - 1 if k < len(dates) else np.nan
            r[f"spy_fwd_{h}"] = spy.iloc[k] / spy.iloc[e - 1] - 1 if k < len(dates) and e >= 1 else np.nan
        rows.append(r)
    return pd.concat([trades.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _grp(df: pd.DataFrame, by, label: str) -> None:
    g = df.groupby(by, observed=True)["ret_net"]
    t = pd.DataFrame({"trades": g.size(), "win_rate": g.apply(lambda x: (x > 0).mean()),
                      "avg_trade": g.mean(), "total": g.sum()})
    print(f"\n--- by {label} ---")
    print(t.to_string(float_format=lambda v: f"{v:+.1%}" if abs(v) < 50 else f"{v:.0f}"))


def report(df: pd.DataFrame, name: str) -> None:
    pd.set_option("display.width", 200)
    n = len(df)
    win = df["ret_net"] > 0
    print(f"\n===== {name}: {n} trades, win rate {win.mean():.0%}, avg {df['ret_net'].mean():+.2%}, "
          f"avg win {df.loc[win, 'ret_net'].mean():+.1%}, avg loss {df.loc[~win, 'ret_net'].mean():+.1%} =====")

    # 1. exits
    g = df.groupby("exit")
    print("\n--- 1. how trades ended ---")
    t = pd.DataFrame({"trades": g.size(), "share": (g.size() / n).map("{:.0%}".format),
                      "avg_ret": g["ret_net"].mean().map("{:+.1%}".format),
                      "avg_days": g["days"].mean().map("{:.0f}".format)})
    print(t.to_string())

    # 2. anatomy of the losses
    loss = df[~win]
    stop = df[df["exit"] == "stop line"]
    print("\n--- 2. where the losses come from ---")
    if len(loss):
        print(f"losing trades: {len(loss)}, total loss {loss['ret_net'].sum():+.1%} (sum of trade returns)")
    if len(stop):
        fast = stop["days"] <= 3
        print(f"stopped within 3 days: {fast.sum()} of {len(stop)} stops ({fast.mean():.0%})")
        line = stop["ret_net"].median()
        gapped = stop["ret_net"] < line - 0.01
        print(f"stops that filled more than 1 point below the usual stop result ({line:+.1%}): {gapped.sum()} "
              f"({gapped.mean():.0%}), average {stop.loc[gapped, 'ret_net'].mean() if gapped.any() else 0:+.1%}, "
              f"worst {stop['ret_net'].min():+.1%}")
    gave_back = loss[loss["mfe"] >= 0.10]
    print(f"losers that were up >= +10% at some point first: {len(gave_back)} ({len(gave_back) / max(len(loss), 1):.0%} of losers)")
    never = loss[loss["mfe"] < 0.02]
    print(f"losers that never got more than +2% above entry: {len(never)} ({len(never) / max(len(loss), 1):.0%} of losers)")
    print(f"average entry-day gap {df['entry_gap'].mean():+.2%}; average first-day return {df['day1'].mean():+.2%}")

    # 3. selection vs exits
    print("\n--- 3. were the picks good and the exits bad? (simply holding, no exits, before costs) ---")
    rows = []
    for h in HORIZONS:
        f, s = df[f"fwd_{h}"].dropna(), df[f"spy_fwd_{h}"].reindex(df[f"fwd_{h}"].dropna().index)
        if len(f):
            rows.append({"hold": f"{h} days", "trades": len(f), "avg": f.mean(), "median": f.median(),
                         "up_share": (f > 0).mean(), "spy_avg": s.mean(), "avg_vs_spy": (f - s).mean()})
    print(pd.DataFrame(rows).set_index("hold").to_string(float_format=lambda v: f"{v:+.1%}" if abs(v) < 5 else f"{v:.0f}"))
    print("If holding beats the strategy's average trade, the exits are cutting winners; if holding is also "
          "negative vs SPY, the stock selection itself has no edge.")

    # 4. groups
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    _grp(df, "year", "year")
    df["market"] = np.where(df["spy_trend"] > 0, "SPY above 200-day avg", "SPY below 200-day avg")
    _grp(df, "market", "market regime at entry")
    df["price_bucket"] = pd.cut(df["price"], [0, 5, 10, 20, 50, np.inf], labels=["<$5", "$5-10", "$10-20", "$20-50", ">$50"])
    _grp(df, "price_bucket", "share price")
    df["adv_bucket"] = pd.cut(df["adv_musd"], [0, 2, 5, 20, 100, np.inf],
                              labels=["<$2M", "$2-5M", "$5-20M", "$20-100M", ">$100M"])
    _grp(df, "adv_bucket", "daily dollar volume")
    df["gap_bucket"] = pd.cut(df["entry_gap"], [-np.inf, -0.02, 0.0, 0.02, np.inf],
                              labels=["gap down >2%", "down 0-2%", "up 0-2%", "gap up >2%"])
    _grp(df, "gap_bucket", "entry-day gap")

    # 5. extremes
    cols = ["symbol", "entry_date", "exit_date", "days", "exit", "ret_net", "mfe", "mae", "entry_gap", "price",
            "fwd_60"]
    pct = {"ret_net", "mfe", "mae", "entry_gap", "fwd_60"}
    for title, part in (("10 worst trades", df.nsmallest(10, "ret_net")), ("10 best trades", df.nlargest(10, "ret_net"))):
        show = part[cols].copy().astype(object)
        for c in cols:
            if c in pct:
                show[c] = part[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
            elif c == "price":
                show[c] = part[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        print(f"\n--- {title} ---")
        print(show.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Trade-by-trade analysis of a strategy backtest")
    p.add_argument("name", help="strategy name, e.g. second_wave")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    f = RS.OUT / f"{args.name}_trades.csv"
    if not f.exists() or f.stat().st_size < 5:
        raise SystemExit(f"No trades file {f}: run python run_strategies.py backtest {args.name} first")
    trades = pd.read_csv(f)
    if trades.empty:
        raise SystemExit("The backtest made no trades")
    df = enrich(trades)
    out = RS.OUT / f"{args.name}_trades_detail.csv"
    df.to_csv(out, index=False)
    report(df, args.name)
    print(f"\nEvery trade with these details: {out} (open in Excel, sort by any column)")


if __name__ == "__main__":
    main()
