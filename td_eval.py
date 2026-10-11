#!/usr/bin/env python3
"""
td_eval.py - is the TechDist distribution forecast any good? (step P3)

Scores the walk-forward forecasts of td_model.py on the stocks of a basket (pool A or B), split
into fixed periods:
  dev    2021-01 .. 2023-12   look as often as you like
  val    2024                 choose models / thresholds here
  test   2025-01 .. today     LOCKED. Opened only with --unlock-test; every opening is recorded in
                              reports/td/test_unlocks.csv. Look once, at the very end.

Per model
  CRPS       ~ 2 x mean pinball loss over the 13 quantiles, in basis points of return (lower = better)
  skill      1 - CRPS / CRPS(M0). M0 knows the volatility but has no view on direction or shape
  DM t       Diebold-Mariano: t-stat of the daily loss difference vs M0 (Newey-West, lag h)
  cov50/80/90  share of outcomes inside the central 50 / 80 / 90 % intervals (should match)
  PIT KS     max distance of the forecast probabilities of the outcomes from uniform (0 = calibrated)
  AUC        does P(R > 0) separate up from down outcomes? (0.5 = no)
  Brier      squared error of P(R > 0)
  IC, IC t   mean daily Spearman correlation between the predicted mean and the outcome across the
             basket, and its Newey-West t-stat (the ranking power)
  top3-bot3  mean outcome of the 3 highest minus the 3 lowest predicted means, bp per trade window
Pass rule on val (fixed in advance): best model skill > 0 with DM t > 2. Ranking also needs IC t > 2.

Usage
  python td_eval.py                    # all horizons found, pools B and A, dev + val
  python td_eval.py --h 3 --pool B
  python td_eval.py --unlock-test      # adds the test period (recorded)
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import td_data as T
import td_features as F
import td_model as M

log = logging.getLogger("td_eval")

PERIODS = {"dev": ("2021-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31"),
           "test": ("2025-01-01", "2100-01-01")}
TOP_N = 3


def nw_t(x: np.ndarray, lag: int) -> float:
    """t-stat of the mean of a daily series with Newey-West standard errors."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(fc: pd.DataFrame, h: int) -> pd.DataFrame:
    """One row of metrics per model for the forecast rows given (outcomes must be known)."""
    fc = fc[fc["y"].notna()]
    base = fc[fc["model"] == "M0"].set_index(["date", "symbol"])
    base_loss = pd.Series(M.pinball(base[M.QCOLS].to_numpy(), base["y"].to_numpy()), index=base.index)
    rows = []
    for m, g in fc.groupby("model", sort=False):
        g = g.set_index(["date", "symbol"])
        Q, y = g[M.QCOLS].to_numpy(), g["y"].to_numpy()
        loss = pd.Series(M.pinball(Q, y), index=g.index)
        d = (base_loss.reindex(g.index) - loss).groupby(level="date").mean()
        pit = M.cdf_at(Q, y)
        ecdf = np.sort(pit)
        ks = float(np.max(np.abs(ecdf - (np.arange(1, len(ecdf) + 1) - 0.5) / len(ecdf))))
        gr = g.reset_index()
        ic = gr.groupby("date").apply(lambda x: x["mean"].corr(x["y"], method="spearman")
                                      if len(x) >= 5 else np.nan, include_groups=False)

        def spread(x):
            x = x.sort_values("mean")
            return x["y"].tail(TOP_N).mean() - x["y"].head(TOP_N).mean() if len(x) >= 2 * TOP_N else np.nan
        ls = gr.groupby("date").apply(spread, include_groups=False)
        rows.append(dict(
            model=m, n=len(g), days=g.index.get_level_values("date").nunique(),
            crps_bp=2 * loss.mean() * 1e4,
            skill=1 - loss.mean() / base_loss.reindex(g.index).mean(),
            dm_t=nw_t(d.to_numpy(), h),
            cov50=np.mean((y >= g["q25"]) & (y <= g["q75"])),
            cov80=np.mean((y >= g["q10"]) & (y <= g["q90"])),
            cov90=np.mean((y >= g["q05"]) & (y <= g["q95"])),
            pit_ks=ks,
            auc=auc(g["p_up"].to_numpy(), y > 0),
            brier=float(np.mean((g["p_up"].to_numpy() - (y > 0)) ** 2)),
            ic=float(ic.mean()), ic_t=nw_t(ic.to_numpy(), h),
            ls_bp=float(ls.mean() * 1e4), ls_t=nw_t(ls.to_numpy(), h),
            p_up_p10=float(np.quantile(g["p_up"], 0.1)), p_up_p90=float(np.quantile(g["p_up"], 0.9)),
        ))
    return pd.DataFrame(rows)


def record_unlock(h: int) -> int:
    f = M.REPORT_DIR / "test_unlocks.csv"
    M.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(time=pd.Timestamp.now().isoformat(timespec="seconds"), h=h)]).to_csv(
        f, mode="a", header=not f.exists(), index=False)
    return len(pd.read_csv(f))


FMT = dict(crps_bp="{:.1f}", skill="{:+.2%}", dm_t="{:+.2f}", cov50="{:.0%}", cov80="{:.0%}", cov90="{:.0%}",
           pit_ks="{:.3f}", auc="{:.3f}", brier="{:.4f}", ic="{:+.3f}", ic_t="{:+.2f}", ls_bp="{:+.1f}",
           ls_t="{:+.2f}", p_up_p10="{:.2f}", p_up_p90="{:.2f}")


def show(tab: pd.DataFrame) -> str:
    t = tab.copy()
    for c, f in FMT.items():
        if c in t:
            t[c] = t[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    return t.to_string(index=False)


def evaluate(h: int, pool: str, target: str, unlock: bool) -> pd.DataFrame:
    fc = pd.read_parquet(M.forecast_file(h, target))
    if pool != "all":
        pools = T.load_pools()
        members = pools[pools["pool"] == pool][["date", "symbol"]]
        fc = fc.merge(members, on=["date", "symbol"])
    periods = ["dev", "val"] + (["test"] if unlock else [])
    out = []
    print(f"\n=== horizon {h} day(s), target {target}, pool {pool} ===")
    for per in periods:
        a, b = PERIODS[per]
        x = fc[(fc["date"] >= a) & (fc["date"] <= b)]
        if x["y"].notna().sum() == 0:
            continue
        tab = score(x, h)
        tab.insert(0, "period", per)
        out.append(tab)
        print(f"\n[{per}] {a} .. {min(pd.Timestamp(b), x['date'].max()).date()}")
        print(show(tab.drop(columns=["period"])))
    res = pd.concat(out, ignore_index=True)
    # by-year view of the best non-baseline model on val
    val = res[(res["period"] == "val") & (res["model"] != "M0")]
    if len(val):
        best = val.sort_values("skill", ascending=False).iloc[0]
        x = fc[fc["model"].isin(["M0", best["model"]])]
        if not unlock:
            x = x[x["date"] < PERIODS["test"][0]]
        yrs = []
        for y, g in x.groupby(x["date"].dt.year):
            s = score(g, h)
            s = s[s["model"] == best["model"]]
            if len(s):
                yrs.append(s.assign(year=y)[["year", "skill", "dm_t", "auc", "ic", "ic_t", "ls_bp"]])
        if yrs:
            print(f"\nBy year, {best['model']}:")
            print(show(pd.concat(yrs, ignore_index=True)))
        ok_dist = best["skill"] > 0 and best["dm_t"] > 2
        ok_rank = best["ic_t"] > 2
        print(f"\nVal verdict ({best['model']}): distribution {'PASS' if ok_dist else 'FAIL'} "
              f"(skill {best['skill']:+.2%}, DM t {best['dm_t']:+.2f}); "
              f"ranking {'PASS' if ok_rank else 'FAIL'} (IC t {best['ic_t']:+.2f})")
    M.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(M.REPORT_DIR / f"eval_h{h}{'' if target == 'y' else '_' + target}_{pool}.csv", index=False)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="TechDist forecast evaluation")
    p.add_argument("--h", default=None, help="horizons, e.g. 1,3,5 (default: every forecast file found)")
    p.add_argument("--pool", default="B,A", help="B, A, all, or a comma list")
    p.add_argument("--target", choices=["y", "ex"], default="y")
    p.add_argument("--unlock-test", action="store_true", help="also score the locked test period (recorded)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    hs = [int(x) for x in args.h.split(",")] if args.h else \
        [h for h in F.HORIZONS if M.forecast_file(h, args.target).exists()]
    if not hs:
        raise SystemExit("No forecasts yet: run python td_model.py")
    for h in hs:
        if args.unlock_test:
            n = record_unlock(h)
            print(f"TEST PERIOD UNLOCKED (opening #{n} recorded). Results seen now must not be used to change "
                  f"the models, or the test stops being a test.")
        for pool in args.pool.split(","):
            evaluate(h, pool.strip(), args.target, args.unlock_test)
        M.log_experiment(step="eval", h=h, target=args.target, note="test unlocked" if args.unlock_test else "")


if __name__ == "__main__":
    main()
