# US daily multi-factor strategy (runs locally on Windows, real account traded by hand)

- **Universe**: top 1000 US common stocks by 60-day average dollar volume, refreshed monthly. Delisted stocks are kept in the history, so backtests are not limited to today's survivors.
- **Factors**: about 50 in three groups: price/volume, minute bars, and SEC fundamentals (point-in-time by filing date).
- **Selection**: every January the factors are re-selected automatically, using only data from before that year. The rules are strict: multiple-testing correction, stability across years, profitability after costs, and de-duplication.
- **Two accounts, one score**:

| | Model account (paper) | Real account |
|---|---|---|
| Capital | $100,000 simulated | your own (~$1,000) |
| Holdings | 25 stocks, inverse-volatility weights, ≤ 6 per sector | 5 stocks, equal weights, ≤ 2 per sector |
| Rebalance | every 5 trading days, takes **every** signal | every 21 trading days (monthly) |
| Sell rule | only when rank falls below 60 | only when rank falls below 15 |
| Execution | auto-filled at the next real open, spread + impact charged | you trade on Alpaca, then `record` |
| Folder | `portfolio\model\` | `portfolio\` |

Both accounts use the same rules as `backtest.py` (`PROFILES` in `backtest.py`). Change them there and the live signals follow.

> Personal research only. Model signals are not investment advice. No backtest guarantees future returns.

## Files

| File | Purpose | When |
|---|---|---|
| `data.py` | daily / minute bars, SPY, security master incl. delisted stocks, universe | daily (auto) |
| `fundamentals.py` | SEC EDGAR financials + sectors, point-in-time | weekly (auto, Mondays) |
| `factors.py` | computes ~50 factors | daily (auto) |
| `evaluate.py` | single-factor tests + automatic factor selection | research |
| `model.py` | combines factors into a score (`wf` walk-forward by default) | daily (auto) |
| `backtest.py` | portfolio backtest with costs, vs SPY; defines the two account profiles | research |
| `daily_signal.py` | trade lists for both accounts, model-account fills, HTML email | daily (auto) + `record` after real trades |
| `report.py` | HTML daily report | called by daily_signal |
| `export_site.py` / `app.py` | web page snapshot / Streamlit page | daily (auto) / when you want |
| `check_email.py` | email diagnosis | when email fails |
| `run_daily.bat` / `setup_task.bat` / `publish.bat` | daily job / schedule it / push the web page | |

## First install

1. Install Python 3.11+ (tick **Add Python to PATH**).
2. Put the files in one folder, e.g. `B:\quant`, open a command prompt there and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your keys (Alpaca, Massive, `SEC_USER_AGENT`, email). **Never put real keys in `.env.example`.**
4. Test email: `python check_email.py`, then `python daily_signal.py test-email`.

## First data download (one-off, slow)

```
python data.py update             :: daily bars from 2016 incl. delisted stocks (an older 2020 install back-fills automatically)
python fundamentals.py update     :: SEC filings, ~15-20 minutes
python data.py update --minute    :: optional, minute bars, ~3 hours, resumable
python factors.py
```

## Research: what to look at

```
python evaluate.py --end 2023-12-31        :: factor tests + selection using data up to 2023 only
python model.py                            :: walk-forward scores (re-selects factors each January)
python backtest.py --profile model         :: the $100k, 25-stock weekly book
python backtest.py --profile real          :: the 5-stock monthly book
streamlit run app.py                       :: view everything in the browser
```

The `wf` score is the honest test: each year's factors are chosen without seeing that year. If `wf` does not beat SPY after costs, do not trust a better-looking result from a hand-picked factor list. Useful variations:

```
python backtest.py --profile real --top 8 --rebalance 42      :: fewer trades
python backtest.py --profile model --weighting equal
python model.py --refresh                                     :: redo the yearly selections after changing factors
```

## Strategies: one idea per file (`strategies/`, `engine.py`, `run_strategies.py`)

Every idea lives in its own file in `strategies/` and runs through one engine (`engine.py`), so they can be
compared fairly: orders fill at the next open, costs are charged, and profit / stop / time exits are checked
every day with the day's high and low. The same code produces the backtest and the live paper account.

| File | Strategies | Idea |
|---|---|---|
| `buy_hold.py` | `spy_hold`, `nvda_googl_hold` | buy once, never sell (SPY is the benchmark) |
| `factor_ranking.py` | `factor_weekly25`, `factor_monthly5` | multi-factor ranking from `model.py` scores |
| `stealth.py` | `stealth` | small/mid caps with rising volume and a sideways price; +50% / −20% / 126 days |
| `swing.py` | `swing` | hot stocks after a price/volume burst; +8% / −5% / 10 days |

```
python run_strategies.py list
python run_strategies.py backtest                 :: all strategies, comparison table
python run_strategies.py backtest stealth swing   :: only some
python run_strategies.py live                     :: advance every $100,000 paper account (run_daily.bat does it)
```
The web page's **Strategies** tab shows the comparison, every paper account and each strategy's positions and
trades. run_daily.bat updates the paper accounts every day and re-runs all backtests on Fridays.

**Adding an idea**: copy `strategies/buy_hold.py` to a new file, change `decide()` (what to buy or sell after
each close; attach `tp`, `sl`, `max_days` to a buy for automatic exits), give it a new `name`. It is picked up
automatically. Set `enabled = False` to switch one off.

## Event-driven setups (`event.py`)

A separate way to trade, closer to how a person picks stocks: buy one stock when its setup looks right, then
leave it alone until it hits the take-profit line, the stop-loss line or the maximum holding time.

| Setup | Looks for | Exits | Slots |
|---|---|---|---|
| `stealth` | small/mid caps ($1M–$30M daily volume) with volume rising steadily while the price moves sideways | +50% / −20% / 126 days | 10 |
| `swing` | liquid stocks right after a price or volume burst | +8% / −5% / 10 days | 5 |

A LightGBM model estimates the chance that the profit line is hit before the stop line, trained walk-forward
(each year only on trades whose outcome was already known). The backtest compares the model's picks with
random picks from the same pool and the same exits; the model is only useful if it clearly beats them.

```
python event.py backtest --setup stealth
python event.py backtest --setup swing
python event.py backtest --setup stealth --tp 1.0 --sl 0.25 --max-days 189
python event.py today --setup stealth
```
`today` lists today's candidates with their probability and exit lines, and checks the positions you list in
`portfolio\event_<setup>.csv` (`symbol,entry_date,entry_price,shares`).

## Every day

1. Run `setup_task.bat` once. The job then runs every weekday at 8:30 pm.
2. On a rebalance evening you get an email with what to sell and what to buy.
3. On the next trading day after the open (9:30–9:45), trade on Alpaca: **sell first** (all shares), **then buy** (market order by dollar amount).
4. After the fills, run `python daily_signal.py record`. Press Enter to accept the suggested value.
5. To see holdings, run `python daily_signal.py status`, or open the page.

The real account only rebalances about once a month, so most evenings need nothing from you. The model account keeps trading every week on its own, so it records what following every signal would have done.

`portfolio\holdings.csv` can be edited in Excel (the `CASH` row is cash) if you trade outside the list.

## Web page (Streamlit Community Cloud)

Local preview: `python export_site.py`, then `streamlit run app.py`. To publish: push the folder to GitHub (`.gitignore` already excludes `.env`, `data\`, `portfolio\`, `reports\`, `logs\`), create an app on share.streamlit.io with main file `app.py`, then set `PUBLISH=1` in `run_daily.bat`. Set `SITE_SHOW_REAL_AMOUNTS=0` in `.env` to hide your real dollar amounts.

## TechDist research (tech basket, return distributions)

A separate research line on ~10 large tech / AI stocks, 2020 onwards only: predict the whole
distribution of each stock's next 1-5 day return, then (later) trade the best names with intraday
limit orders. Design, pass rules and build plan: `docs/TECHDIST.md`. Steps P0-P3 are built:

```
python td_data.py update          :: own data folder data\td\ (daily + minute bars, ~15-20 min first time)
python td_features.py --check     :: look-ahead test, must print OK three times
python td_features.py
python td_model.py                :: walk-forward models M0 / M1 / M2 / ENS, horizons 1, 3, 5
python td_eval.py                 :: forecast quality on dev (2021-23) and val (2024); test stays locked
```

## FAQ

- **No email**: check today's log in `logs\`; run `python check_email.py`.
- **Change the score method**: change `METHOD` in `run_daily.bat` and `SCORE_METHOD` in `.env` together.
- **Change holdings / rebalance frequency**: edit `PROFILES` in `backtest.py`, then re-run the backtest.
- **Remove the scheduled task**: `schtasks /Delete /TN QuantDaily /F`
