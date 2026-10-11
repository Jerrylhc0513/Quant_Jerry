#!/usr/bin/env python3
"""
td_model.py - TechDist return-distribution models, walk-forward (step P2)

For every stock and every day the models output the whole distribution of the next-h-day return
(entry at the next open), as 13 quantiles, plus its mean and P(return > 0).

  1. Volatility first (HAR): log realised variance over the label window is regressed on today's
     daily / weekly / monthly realised variance, QQQ's monthly variance and the earnings flag.
     sigma_hat = that forecast. Every return is divided by sigma_hat, so all stocks and calm /
     stormy periods share one standardised scale z = R / sigma_hat.
  2. Then the shape and the centre of z:
       M0   unconditional: the empirical quantiles of past z. No view on direction. The benchmark.
       M1   ridge regression for the mean of z, plus the empirical quantiles of its residuals.
       M2   LightGBM quantile regression, one model per quantile level, then sorted.
       ENS  average of the M1 and M2 quantiles.
     A model is only worth keeping if it beats M0 out of sample (td_eval.py).

Walk-forward: refit on the first trading day of every month from FIRST_FORECAST. A refit only
trains on rows whose label was complete before that day (no overlap with the future), and
forecasts every day until the next refit. Training data start in 2020 (SAMPLE_START).

Usage
  python td_model.py                    # horizons 1, 3, 5; all models
  python td_model.py --horizons 3       # one horizon (quicker)
  python td_model.py --target ex        # predict the QQQ-neutral return instead (for ranking)
Output: data/td/forecasts_h{h}[_ex].parquet ; every run is logged in reports/td/experiments.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import td_data as T
import td_features as F

log = logging.getLogger("td_model")

QUANTILES = np.array([0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95])
QCOLS = [f"q{int(round(t * 100)):02d}" for t in QUANTILES]
FIRST_FORECAST = "2021-01-01"      # one year of training data first
REFIT_MONTHS = 1
Z_CLIP = 8.0
MODELS = ("M0", "M1", "M2", "ENS")
VOL_COLS = ["rv_d", "rv_w", "rv_m", "qqq_rv_m", "earn_soon"]
X_COLS = [c for c in F.FEATURES if c != "sym_id"]     # pooled model, no stock identity
RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0)
LGB_PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=7, min_data_in_leaf=300,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, seed=7, num_threads=0)
LGB_ROUNDS = 200
REPORT_DIR = Path(__file__).resolve().parent / "reports" / "td"


def forecast_file(h: int, target: str = "y") -> Path:
    return T.TD_DIR / f"forecasts_h{h}{'' if target == 'y' else '_' + target}.parquet"


# ============================== Distribution helpers ==============================
def cdf_at(Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P(R <= x) per row from its quantiles: linear between levels, capped at 0.025 / 0.975 beyond
    the 5% / 95% quantiles (the models cannot resolve the far tails)."""
    Q = np.sort(Q, axis=1)
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n, K = Q.shape
    k = (Q <= x).sum(axis=1)                                  # levels at or below x
    lo, hi = np.clip(k - 1, 0, K - 1), np.clip(k, 0, K - 1)
    r = np.arange(n)
    ql, qh = Q[r, lo], Q[r, hi]
    gap = qh - ql
    frac = np.where(gap > 0, (x[:, 0] - ql) / np.where(gap > 0, gap, 1.0), 0.0)
    tau = QUANTILES[lo] + frac * (QUANTILES[hi] - QUANTILES[lo])
    return np.where(k == 0, 0.025, np.where(k == K, 0.975, tau))


def dist_stats(Q: np.ndarray, thresholds=(0.0,)) -> dict:
    """Quantile matrix (rows x 13 levels) -> mean and P(R > c) for each threshold c."""
    Q = np.sort(Q, axis=1)
    w = np.diff(np.r_[0.0, (QUANTILES[1:] + QUANTILES[:-1]) / 2, 1.0])      # mass around each level
    out = {"mean": Q @ w}
    for c in thresholds:
        out[f"p_gt_{c:g}"] = 1.0 - cdf_at(Q, np.full(len(Q), c))
    return out


def pinball(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row average pinball loss over the 13 levels; 2x this approximates the CRPS."""
    u = y[:, None] - Q
    return (np.maximum(QUANTILES * u, (QUANTILES - 1) * u)).mean(axis=1)


# ============================== Volatility (HAR) ==============================
class HarVol:
    def fit(self, X: pd.DataFrame, rvf: np.ndarray):
        A = self._design(X)
        ly = np.log(rvf + 1e-10)
        ok = np.isfinite(ly) & np.isfinite(A).all(axis=1)
        self.coef, *_ = np.linalg.lstsq(A[ok], ly[ok], rcond=None)
        self.s2 = float(np.var(ly[ok] - A[ok] @ self.coef))
        return self

    def _design(self, X):
        A = X[VOL_COLS].astype(float).copy()
        A["earn_soon"] = A["earn_soon"].fillna(0)
        A = A.fillna(A.median())
        return np.c_[np.ones(len(A)), A.to_numpy()]

    def sigma(self, X):
        return np.sqrt(np.exp(self._design(X) @ self.coef + 0.5 * self.s2))


# ============================== Mean / shape models ==============================
class Prep:
    """Median fill + standardisation, fitted on the training rows only."""

    def fit(self, X: pd.DataFrame):
        self.med = X.median()
        Z = X.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1).fillna(1)
        return self

    def __call__(self, X):
        return ((X.fillna(self.med) - self.mu) / self.sd).fillna(0).to_numpy()


def ridge_fit(A: np.ndarray, z: np.ndarray, lam: float) -> np.ndarray:
    A1 = np.c_[np.ones(len(A)), A]
    R = lam * np.eye(A1.shape[1])
    R[0, 0] = 0.0
    return np.linalg.solve(A1.T @ A1 + R, A1.T @ z)


def ridge_predict(A, b):
    return np.c_[np.ones(len(A)), A] @ b


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, h: int) -> dict[str, np.ndarray]:
    """All models on one walk-forward fold. Returns {model: quantiles of R (rows x 13)} and sigma."""
    vol = HarVol().fit(train, train[f"rvf_h{h}"].to_numpy())
    s_tr, s_te = vol.sigma(train), vol.sigma(test)
    z = np.clip(train[target].to_numpy() / s_tr, -Z_CLIP, Z_CLIP)
    out = {"sigma": s_te}

    out["M0"] = np.tile(np.quantile(z, QUANTILES), (len(test), 1))

    prep = Prep().fit(train[X_COLS])
    A_tr, A_te = prep(train[X_COLS]), prep(test[X_COLS])
    dates = train["date"].to_numpy()
    cut = np.quantile(dates.astype("datetime64[D]").astype(float), 0.8)
    inner = dates.astype("datetime64[D]").astype(float) < cut
    best, best_mse = RIDGE_GRID[0], np.inf
    for lam in RIDGE_GRID:
        b = ridge_fit(A_tr[inner], z[inner], lam * inner.sum() / 1000)
        mse = float(np.mean((ridge_predict(A_tr[~inner], b) - z[~inner]) ** 2))
        if mse < best_mse:
            best, best_mse = lam, mse
    b = ridge_fit(A_tr, z, best * len(z) / 1000)
    resid_q = np.quantile(z - ridge_predict(A_tr, b), QUANTILES)
    out["M1"] = ridge_predict(A_te, b)[:, None] + resid_q[None, :]
    out["ridge_lambda"] = best

    Xtr, Xte = train[X_COLS].to_numpy(float), test[X_COLS].to_numpy(float)
    ds = lgb.Dataset(Xtr, z, free_raw_data=False)
    q2 = np.column_stack([lgb.train({**LGB_PARAMS, "alpha": float(t)}, ds, LGB_ROUNDS).predict(Xte)
                          for t in QUANTILES])
    out["M2"] = np.sort(q2, axis=1)
    out["ENS"] = (out["M1"] + out["M2"]) / 2
    return out


# ============================== Walk-forward ==============================
def refit_dates(dates: pd.DatetimeIndex, months: int) -> list[pd.Timestamp]:
    dates = dates[dates >= pd.Timestamp(FIRST_FORECAST)]
    firsts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    return list(firsts.iloc[::months])


def run(h: int, target: str = "y", months: int = REFIT_MONTHS) -> pd.DataFrame:
    df = F.load_features()
    tcol = f"{target}_h{h}"
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    refits = refit_dates(dates, months)
    frames = []
    t0 = time.time()
    for i, T0 in enumerate(refits):
        T1 = refits[i + 1] if i + 1 < len(refits) else dates[-1] + pd.Timedelta(days=1)
        train = df[(df[f"end_h{h}"] < T0) & df[tcol].notna() & df[f"rvf_h{h}"].notna()]
        test = df[(df["date"] >= T0) & (df["date"] < T1)]
        if len(test) == 0 or train["date"].nunique() < 200:
            continue
        res = fit_predict(train, test, tcol, h)
        for m in MODELS:
            Qr = res[m] * res["sigma"][:, None]
            st = dist_stats(Qr, thresholds=(0.0,))
            f = pd.DataFrame(Qr, columns=QCOLS)
            f.insert(0, "model", m)
            f.insert(1, "date", test["date"].to_numpy())
            f.insert(2, "symbol", test["symbol"].to_numpy())
            f["sigma"] = res["sigma"]
            f["mean"] = st["mean"]
            f["p_up"] = st["p_gt_0"]
            f["y"] = test[tcol].to_numpy()
            f["refit"] = T0
            frames.append(f)
        if (i + 1) % 6 == 0 or i + 1 == len(refits):
            el = time.time() - t0
            log.info("h=%d %s: refit %d/%d (%s), %.0f s, about %.0f s left", h, target, i + 1, len(refits),
                     T0.date(), el, el / (i + 1) * (len(refits) - i - 1))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(forecast_file(h, target), index=False)
    return out


def log_experiment(step: str, h: int, target: str, config: str = "", note: str = "") -> None:
    """Every model run and every evaluation is one line, so the number of tries stays visible."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    f = REPORT_DIR / "experiments.csv"
    row = pd.DataFrame([dict(time=pd.Timestamp.now().isoformat(timespec="seconds"), step=step, h=h,
                             target=target, config=config, note=note)])
    row.to_csv(f, mode="a", header=not f.exists(), index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="TechDist walk-forward distribution models")
    p.add_argument("--horizons", default=",".join(map(str, F.HORIZONS)))
    p.add_argument("--target", choices=["y", "ex"], default="y")
    p.add_argument("--refit-months", type=int, default=REFIT_MONTHS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    config = dict(features=X_COLS, vol=VOL_COLS, lgb=LGB_PARAMS, rounds=LGB_ROUNDS, ridge=RIDGE_GRID,
                  quantiles=QUANTILES.tolist(), first=FIRST_FORECAST, refit=args.refit_months)
    cfg_hash = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
    for h in [int(x) for x in args.horizons.split(",")]:
        out = run(h, args.target, args.refit_months)
        log.info("h=%d: %d forecast rows, %s to %s -> %s", h, len(out), out["date"].min().date(),
                 out["date"].max().date(), forecast_file(h, args.target).name)
        log_experiment("model", h, args.target, cfg_hash,
                       f"{len(X_COLS)} features, refit every {args.refit_months} month(s)")


if __name__ == "__main__":
    main()
