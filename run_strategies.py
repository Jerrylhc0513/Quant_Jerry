#!/usr/bin/env python3
"""
run_strategies.py - run every strategy in strategies/ through the same engine and compare them.

  python run_strategies.py list                         what is there
  python run_strategies.py backtest                     backtest all enabled strategies, print a comparison
  python run_strategies.py backtest stealth spy_hold    only some
  python run_strategies.py live                         advance every paper account to today (run_daily.bat)
  python run_strategies.py split A B --cut 2022-01-01   train period before the cut, test period after it
  python run_strategies.py fills A [--detail]           every order A executed in its backtest

Backtests: reports/strategies/<name>_nav.csv, <name>_trades.csv, summary.csv
Paper accounts ($100,000 each, every signal filled at the next open with costs):
  portfolio/strategies/<name>/  book.json, history.csv, trades.csv
A paper account starts the first time `live` runs, so its record is genuinely out-of-sample.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import engine as EN
import evaluate as E
from strategies import discover

OUT = E.REPORT_DIR / "strategies"
LIVE_DIR = Path(os.environ.get("PORTFOLIO_DIR", Path(__file__).resolve().parent / "portfolio")) / "strategies"
log = logging.getLogger("strategies")


def _pick(names: list[str] | None) -> dict:
    allst = discover(include_disabled=bool(names))
    if not names:
        return {k: v for k, v in allst.items() if v.enabled}
    bad = [n for n in names if n not in allst]
    if bad:
        raise SystemExit(f"Unknown strategies: {bad}. Available: {list(allst)}")
    return {n: allst[n] for n in names}


def _trade_stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    w = tr["ret_net"] > 0
    return {"trades": len(tr), "win_rate": w.mean(), "avg_trade": tr["ret_net"].mean(),
            "avg_days": tr["days"].mean()}


def cmd_list() -> None:
    for name, st in discover(include_disabled=True).items():
        print(f"{name:<18} {'on ' if st.enabled else 'off'}  {st.title} - {st.description}")


def cmd_backtest(names, start=None) -> pd.DataFrame:
    sts = _pick(names)
    md = EN.MarketData()
    spy = md.C["SPY"] if "SPY" in md.C else None
    qqq = md.C["QQQ"] if "QQQ" in md.C else None
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, st in sts.items():
        t0 = time.time()
        if start:
            st.start = start
        try:
            book = EN.backtest(st, md)
        except SystemExit as e:
            log.warning("%s skipped: %s", name, e)
            continue
        hist = pd.DataFrame(book.history)
        if hist.empty:
            log.warning("%s produced no history", name)
            continue
        nav = hist.set_index(pd.to_datetime(hist["date"]))["nav"]
        tr = pd.DataFrame(book.trades)
        hist.assign(spy=spy.reindex(nav.index).to_numpy() if spy is not None else np.nan).to_csv(
            OUT / f"{name}_nav.csv", index=False)
        tr.to_csv(OUT / f"{name}_trades.csv", index=False)
        pd.DataFrame(book.fills).to_csv(OUT / f"{name}_fills.csv", index=False)
        m = EN.metrics(nav, spy, qqq)
        rows.append({"strategy": name, "title": st.title, "start": nav.index[0].date(), "end": nav.index[-1].date(),
                     **m, **_trade_stats(tr)})
        log.info("%s done in %.0f s", name, time.time() - t0)
    summ = pd.DataFrame(rows).set_index("strategy")
    old = OUT / "summary.csv"
    if names and old.exists():                       # keep rows of strategies not run this time
        prev = pd.read_csv(old, index_col=0)
        summ = pd.concat([prev.drop(index=summ.index, errors="ignore"), summ])
    summ.to_csv(old)
    _print(summ)
    return summ


def _print(summ: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    show = summ.copy()
    pct = ["ann_return", "ann_vol", "max_drawdown", "spy_ann", "qqq_ann", "excess_ann", "win_rate", "avg_trade"]
    for c in show.columns:
        if c in pct:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
        elif c in ("sharpe", "calmar"):
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        elif c in ("avg_days",):
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    cols = [c for c in ["title", "start", "end", "ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar", "spy_ann",
                        "qqq_ann", "excess_ann", "trades", "win_rate", "avg_trade", "avg_days"] if c in show]
    print("\n===== Strategy comparison (backtests, net of costs; spy_ann / qqq_ann = SPY / QQQ over the same period) =====")
    print(show[cols].to_string())
    print("\nOnly walk-forward strategies (factor wf scores, stealth, swing) are out-of-sample. Buy-and-hold of names "
          "picked today carries hindsight. Paper accounts (`live`) are the fair test going forward.")


def cmd_live(names) -> None:
    sts = _pick(names)
    md = EN.MarketData()
    failed = []
    for name, st in sts.items():
        t0 = time.time()
        try:
            book = EN.live(st, md, LIVE_DIR / name)
        except SystemExit as e:
            log.warning("%s skipped: %s", name, e)
            continue
        except Exception:                            # one broken strategy must not stop the others
            log.exception("%s failed", name)
            failed.append(name)
            continue
        pend = ", ".join(f"{'SELL' if o.weight == 0 else 'BUY'} {o.symbol}" for o in book.pending) or "none"
        log.info("%s: NAV $%s, %d positions, orders for the next open: %s (%.0f s)", name,
                 f"{book.nav:,.0f}", len(book.positions), pend, time.time() - t0)
    if failed:
        raise SystemExit(f"Strategies that failed: {failed} (the others were updated)")


def cmd_split(names, cut: str, start=None, end=None) -> pd.DataFrame:
    """Backtest each strategy separately on a training period (before `cut`) and a test period (from `cut`).
    Choose between strategies using the training columns only; the test columns show whether the choice holds."""
    sts = _pick(names)
    md = EN.MarketData()
    spy = md.C["SPY"] if "SPY" in md.C else None
    qqq = md.C["QQQ"] if "QQQ" in md.C else None
    cut_ts = pd.Timestamp(cut)
    rows = []
    for name, st in sts.items():
        try:
            st.prepare(md)
        except SystemExit as e:
            log.warning("%s skipped: %s", name, e)
            continue
        row = {"strategy": name}
        for tag, a, b in (("train", start, cut_ts - pd.Timedelta(days=1)), ("test", cut_ts, end)):
            book = EN.backtest(st, md, start=a, end=b, prepared=True)
            hist = pd.DataFrame(book.history)
            tr = pd.DataFrame(book.trades)
            if hist.empty:
                continue
            nav = hist.set_index(pd.to_datetime(hist["date"]))["nav"]
            m = EN.metrics(nav, spy, qqq)
            ts = _trade_stats(tr)
            row.update({f"{tag}_ann": m.get("ann_return"), f"{tag}_spy": m.get("spy_ann"), f"{tag}_qqq": m.get("qqq_ann"),
                        f"{tag}_maxdd": m.get("max_drawdown"), f"{tag}_trades": ts.get("trades", 0),
                        f"{tag}_win": ts.get("win_rate"), f"{tag}_avg_trade": ts.get("avg_trade")})
        rows.append(row)
        log.info("%s done", name)
    out = pd.DataFrame(rows).set_index("strategy")
    show = out.copy().astype(object)
    for c in out.columns:
        if not c.endswith("_trades"):
            show[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    pd.set_option("display.width", 220)
    print(f"\n===== Training period (before {cut_ts.date()}) vs test period (from {cut_ts.date()}), net of costs =====")
    print(show.to_string())
    print("\nPick using the train_* columns only. The choice is believable if its test_* columns are also good.")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "split_summary.csv")
    return out


def cmd_fills(names, start=None, detail: bool = False) -> None:
    """Every order a strategy executed in its backtest: counts per year, each trading day, optionally each fill."""
    sts = _pick(names)
    md = EN.MarketData()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 2000)
    for name, st in sts.items():
        if start:
            st.start = start
        book = EN.backtest(st, md)
        f = pd.DataFrame(book.fills)
        f.to_csv(OUT / f"{name}_fills.csv", index=False)
        print(f"\n===== {name}: {st.title} =====")
        if f.empty:
            print("no orders executed")
            continue
        hist = pd.DataFrame(book.history).set_index("date")
        expo = 1 - hist["cash"] / hist["nav"]
        f["year"] = f["date"].str[:4]
        cost = f["usd"].sum() * st.cost_bps / 1e4
        nav_avg = hist.groupby(hist.index.str[:4])["nav"].mean()
        per_year = f.groupby("year").agg(orders=("symbol", "size"), trade_days=("date", "nunique"),
                                         traded_usd=("usd", "sum"))
        per_year["turnover"] = per_year["traded_usd"] / nav_avg.reindex(per_year.index)
        print(f"{len(f)} orders on {f['date'].nunique()} days; actions {f['action'].value_counts().to_dict()}; "
              f"total traded ${f['usd'].sum():,.0f}, costs about ${cost:,.0f} at {st.cost_bps:g} bp per side")
        print("\nPer year (turnover = traded dollars / average account value):")
        print(per_year.assign(traded_usd=per_year["traded_usd"].map("${:,.0f}".format),
                              turnover=per_year["turnover"].map("{:.0%}".format)).to_string())
        days = f.groupby("date").agg(orders=("symbol", "size"),
                                     buys=("usd", lambda x: x[f.loc[x.index, "action"].isin(["BUY", "ADD"])].sum()),
                                     sells=("usd", lambda x: x[f.loc[x.index, "action"].isin(["SELL", "REDUCE"])].sum()),
                                     reason=("reason", lambda x: x.mode().iloc[0]))
        days["invested_after"] = expo.reindex(days.index)
        days["nav_after"] = hist["nav"].reindex(days.index)
        print("\nEvery trading day:")
        print(days.assign(buys=days["buys"].map("${:,.0f}".format), sells=days["sells"].map("${:,.0f}".format),
                          invested_after=days["invested_after"].map("{:.0%}".format),
                          nav_after=days["nav_after"].map("${:,.0f}".format)).to_string())
        if detail:
            print("\nEvery order:")
            print(f.drop(columns=["year"]).to_string(index=False))
        print(f"\nFull list: {OUT / (name + '_fills.csv')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run and compare the strategies in strategies/")
    p.add_argument("cmd", choices=["list", "backtest", "live", "split", "fills"])
    p.add_argument("names", nargs="*", help="strategy names (default: all enabled)")
    p.add_argument("--start", help="backtest: first signal date for every strategy")
    p.add_argument("--cut", default="2022-01-01", help="split: first day of the test period")
    p.add_argument("--detail", action="store_true", help="fills: also print every single order")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "backtest":
        cmd_backtest(args.names, args.start)
    elif args.cmd == "fills":
        cmd_fills(args.names, args.start, args.detail)
    elif args.cmd == "split":
        cmd_split(args.names, args.cut, args.start)
    else:
        cmd_live(args.names)


if __name__ == "__main__":
    main()
