# TechDist: return-distribution forecasts for a tech basket, with intraday execution

Status: P0–P3 (data, features, models, forecast evaluation) are implemented. P4–P7 are designed
but not built yet. They will only be built if P3 passes.

## 0. Goal and pass rules (fixed before looking at results)

| Item | Setting |
|---|---|
| Accounts | $100k paper account takes every signal. ~$1k real account trades only the top 1–2 names, in a cash account (no PDT rule), fractional shares, no shorts, holding 1–5 days |
| Benchmarks | (1) equal-weight buy-and-hold of the same basket, the one that matters; (2) QQQ |
| Data | 2020 onwards only. 2019 bars are used solely as warm-up for moving averages |
| Go-live rule | Test period: (1) the distribution forecast beats M0 (skill > 0, DM t > 2), and (2) information ratio vs the equal-weight basket > 0.3, or the same return with a clearly smaller drawdown. Otherwise no real money |

## 1. Universe

- **Pool A (fixed, chosen with hindsight)**: AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO AMD ORCL.
- **Pool B (point in time)**: each January, the 10 names with the largest 60-day dollar volume out of
  26 candidates. The candidates are large US tech / internet names that already existed in 2019.
- Models train on all 26 candidates. Evaluation is restricted to a pool.
- If A beats B by a lot, the edge came from picking the winners, not from the model.

## 2. Architecture

```
L0 regime / risk switch -> L1 return distribution (core) -> L2 decisions and sizing
                                                                     |
                              L4 monitoring and kill switch <- L3 intraday execution
```

### L1: return distribution (td_features.py, td_model.py)

- **Timing**: decide after the close of day t, enter at the open of t+1, exit at the close of t+h.
- **Label**: R = log(close[t+h] / open[t+1]), h ∈ {1, 3, 5}.
  - Variant: the QQQ-neutral return `ex` (R minus β times QQQ's return), used for ranking.
- **Volatility first**: a HAR regression of log realised variance on daily, weekly and monthly
  realised variance (minute-based where available), plus QQQ's variance and the earnings flag.
- **Standardise**: z = R / σ̂, so every stock and every period share one scale.
- **Model ladder** (keep a model only if it beats the one below it out of sample):
  - M0: unconditional quantiles of z (the benchmark).
  - M1: ridge regression for the mean of z, plus residual quantiles.
  - M2: LightGBM quantile regression at 13 levels.
  - ENS: average of M1 and M2.
  - Later, only if it adds value: a small MLP/GRU with a Student-t head trained on CRPS (the IVE head).
- **Walk-forward**: monthly refit. Each refit trains only on labels that were complete before the
  refit date. Forecasts start in 2021.
- **Outputs per stock per day**: 13 quantiles, σ̂, mean, P(R > 0).

### Features (55)

| Group | Features |
|---|---|
| Price | returns over 1–60 days; distance to the 5/20/50/200-day averages in σ units; deviation from the 20-day VWAP; up and down streaks; 20-day drawdown; run-up then pullback; close location; gap; intraday return |
| Volume | relative volume; volume trend; up-volume vs down-volume; last-30-minute volume share; Amihud illiquidity |
| Volatility | realised variance (daily / weekly / monthly); realised skew; jump share; first and last 30-minute returns |
| Cross-asset | β to QQQ; residual returns; rank within the basket; QQQ / SMH / TLT trends; QQQ volatility; average correlation |
| Events | days since and to earnings; FOMC / CPI on the next day (from `macro_events.csv`, optional); weekday; days to month end |

`python td_features.py --check` rebuilds everything on data cut at three dates. Every feature must
match the full build exactly, otherwise there is look-ahead.

### Periods

| Period | Dates | Use |
|---|---|---|
| dev | 2021–2023 | free to look |
| val | 2024 | choose models and thresholds |
| test | 2025 → today | locked; `td_eval.py --unlock-test` records every opening |

## 3. L2: decisions and sizing (P4, not built)

- **Score**: S = (E[R] − cost) / σ̂.
- **Buy**: P(R > cost) ≥ θ_buy, S > 0, regime switch on, no earnings within 2 days.
- **Sell**: P(R < 0) ≥ θ_sell, S < −s₀, or take-profit / stop / time exit.
- **Hold**: anything in between (hysteresis).
- The θ values are chosen on val only.
- **Sizing**:
  - ¼ Kelly (w ∝ μ/σ²), at most 30% per name, at most 5 names.
  - Scale the book to about 20% predicted annual volatility.
  - QQQ below its 200-day average → 50% exposure; plus volatility in its top decile → 25%.
- **Core + satellite variant**: 50% held as the equal-weight basket, 50% traded by the model.
- **Research-only variant**: long the top 3 / short the bottom 3, on paper only, to test ranking power.

## 4. L3: intraday execution (P5, not built)

- The IVE volatility and volume models give tomorrow's minute-by-minute σ and volume curve.
- **Each order gets**:
  - A time window: skip the first 5 minutes; prefer 10:00–11:30 and 15:00–15:50.
  - A limit price at a quantile of that window's low.
  - Take-profit at the h-day 80% quantile, stop-loss at the 10% quantile or 1.5σ.
- **Open-gap rules**:
  - Gap up of more than k·σ: skip the order or halve it.
  - Gap down: re-score using the open price.
- **Fallback**: not filled by 15:45 → buy at the close or cancel.
- **Scored against** the open, VWAP and close fills, in bp, including unfilled orders.
- **Adverse selection test**: compare later returns of filled and unfilled orders.
- **Expected gain**: 0–5 bp per trade. The causal VWAP result in the IVE dissertation was also near zero.

## 5. Costs and backtest rules

- **Costs**: 5 bp per side as the base case; 10 bp per side as the stress case.
- **Information**: decisions only use data up to the close of t; fills only use prices after the decision.
- **Limit fills**: an order fills only when the price trades through the limit (low < limit).
- **Two backtests**: a daily backtest for the signal (engine.py) and a minute-level simulation for execution.

## 6. Metrics

- **Forecast** (td_eval.py):
  - CRPS (≈ 2 × mean pinball loss) and skill vs M0.
  - Diebold-Mariano t, Newey-West with lag h.
  - Coverage of the 50 / 80 / 90% intervals; PIT KS distance.
  - AUC and Brier score of P(R > 0).
  - Daily rank IC and its t-stat; top-3 minus bottom-3 spread.
  - Every metric by year.
- **Strategy (P4)**:
  - CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, turnover, hit rate, win/loss ratio.
  - α / β vs the basket and QQQ with Newey-West t; information ratio.
  - Deflated Sharpe ratio, using the number of tries in `reports/td/experiments.csv`.
  - Probability of backtest overfitting; date-block bootstrap.

## 7. Expected results (priors, not promises)

| Part | Realistic expectation |
|---|---|
| σ forecast | strong: large skill vs an unconditional distribution, +1–3% vs plain HAR |
| Direction (μ) | weak: AUC 0.51–0.54; P(R > 0) mostly 0.45–0.58; rank IC 0.01–0.04 |
| Execution | 0–5 bp per trade |
| Most likely strategy result | about the basket's return with a smaller drawdown (from volatility targeting and the trend switch) |
| If direction fails on test | fall back to "hold the basket + risk overlay" |

## 8. Build plan

| Step | Files | Pass rule |
|---|---|---|
| P0 data | `td_data.py` | status report looks complete |
| P1 features | `td_features.py` | `--check` passes |
| P2 models | `td_model.py` | |
| P3 forecast evaluation | `td_eval.py` | on val, best model skill > 0 and DM t > 2. **Stop here otherwise** |
| P4 daily strategy | `strategies/tech_dist.py` | val rules met, then the test run exactly once |
| P5 execution | `exec_model.py`, `exec_sim.py` | beats open / VWAP fills |
| P6 live | `daily_signal.py`, site | |
| P7 monitoring | `monitor.py` | stop at 15% drawdown or broken coverage |

## 9. Commands

```
python td_data.py update          :: first run ~15-20 minutes (minute bars), later runs a few minutes
python td_data.py status
python td_features.py --check     :: look-ahead test, must print OK three times
python td_features.py
python td_model.py                :: h = 1, 3, 5; about 5 minutes per horizon
python td_eval.py                 :: dev + val for pools B and A
python td_model.py --target ex --horizons 3 && python td_eval.py --target ex --h 3
```

Optional inputs in `data\td\`:
- `earnings.csv` (`symbol,date`): real earnings reaction days.
- `macro_events.csv` (`date,event` with event `FOMC` or `CPI`).
