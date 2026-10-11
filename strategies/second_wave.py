"""
second_wave.py - a stock that once exploded, has since come back down, and now shows volume building up
while the price moves sideways: buy, hoping for a second wave.

Rules (all parameters below can be changed)
  1. history: at some point the close was >= SURGE_GAIN above the lowest close of the previous SURGE_DAYS days
     (default +100% within 15 trading days)
  2. pulled back: the price is now >= PULLBACK below the peak reached after that surge, and the surge ended at
     least MIN_DAYS_SINCE trading days ago
  3. volume building: in the last VOL_DAYS days, volume rose versus the day before on at least VOL_UP_DAYS days
     (default: 8 of 10, i.e. at most two days lower than the day before)
  4. price sideways: over the last VOL_DAYS days the price moved at most FLAT_RET and its high/low range is at
     most FLAT_RANGE
  5. tradable: price >= MIN_PRICE, 60-day average dollar volume >= MIN_ADV
  Exits: take-profit TP, stop-loss SL, at most MAX_DAYS trading days.

second_wave_novol is the same without rule 3: if the full version is not clearly better, the volume rule adds
nothing. Everything only uses data up to the signal day.
"""
import numpy as np
import pandas as pd

from engine import Order, Strategy


class SecondWave(Strategy):
    def __init__(self, name, title, use_volume=True, *, surge_gain=1.0, surge_days=15, pullback=0.40,
                 min_days_since=60, vol_days=10, vol_up_days=8, flat_ret=0.05, flat_range=0.12, min_price=2.0,
                 min_adv=1e6, tp=0.50, sl=0.15, max_days=120, max_pos=10, max_new=2, cost_bps=20.0,
                 start="2017-01-03"):
        self.name, self.title = name, title
        self.use_volume = use_volume
        self.p = dict(surge_gain=surge_gain, surge_days=surge_days, pullback=pullback, min_days_since=min_days_since,
                      vol_days=vol_days, vol_up_days=vol_up_days, flat_ret=flat_ret, flat_range=flat_range,
                      min_price=min_price, min_adv=min_adv)
        self.tp, self.sl, self.max_days = tp, sl, max_days
        self.max_pos, self.max_new = max_pos, max_new
        self.cost_bps, self.start = cost_bps, start
        vol = f"volume up on >= {vol_up_days} of {vol_days} days, " if use_volume else ""
        self.description = (f"Once +{surge_gain:.0%} within {surge_days} days, now >= {pullback:.0%} below that peak; "
                            f"{vol}price flat (<= {flat_ret:.0%} move, <= {flat_range:.0%} range over {vol_days} days); "
                            f"exits +{tp:.0%} / -{sl:.0%} / {max_days} days")

    def prepare(self, md):
        p = self.p
        C = md.C.drop(columns=["SPY"], errors="ignore")
        V = md.V[C.columns]
        Cn = C.to_numpy(dtype="float64")
        T, N = Cn.shape
        # 1. surge days and the peak after the most recent surge
        low_n = C.rolling(p["surge_days"], min_periods=p["surge_days"]).min().to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            surge = Cn / low_n - 1 >= p["surge_gain"]
        peak = np.full((T, N), np.nan)
        last_surge = np.full(N, -10 ** 6)
        cur = np.full(N, np.nan)
        for t in range(T):
            c = Cn[t]
            s = surge[t]
            new_episode = s & (t - last_surge > p["surge_days"])
            cur = np.where(new_episode, c, cur)                 # a new surge episode restarts the peak
            cur = np.where(np.isfinite(cur), np.fmax(cur, c), cur)
            last_surge = np.where(s, t, last_surge)
            peak[t] = cur
        since = np.arange(T)[:, None] - np.maximum.accumulate(np.where(surge, np.arange(T)[:, None], -10 ** 6), axis=0)
        with np.errstate(invalid="ignore"):
            pulled = (Cn <= peak * (1 - p["pullback"])) & (since >= p["min_days_since"])
        # 3. volume building up
        n = p["vol_days"]
        up = (V > V.shift(1)).astype("float32").rolling(n, min_periods=n).sum().to_numpy()
        vol_ok = up >= p["vol_up_days"]
        # 4. price sideways
        ret_n = (C / C.shift(n) - 1).abs().to_numpy()
        rng = (C.rolling(n, min_periods=n).max() / C.rolling(n, min_periods=n).min() - 1).to_numpy()
        flat = (ret_n <= p["flat_ret"]) & (rng <= p["flat_range"])
        # 5. tradable
        ok = ((md.P[C.columns] >= p["min_price"]) & (md.ADV[C.columns] >= p["min_adv"])).to_numpy()
        sig = pulled & flat & ok & (vol_ok if self.use_volume else True)
        # rank same-day signals: steadier volume build-up first, then deeper pullback
        vol_ratio = (V.rolling(5).mean() / V.rolling(60, min_periods=40).mean()).to_numpy()
        self.signals = {}
        ti, ni = np.nonzero(sig)
        for t, j in zip(ti, ni):
            self.signals.setdefault(int(t), []).append((float(np.nan_to_num(vol_ratio[t, j])), C.columns[j]))
        self.n_signals = len(ti)

    def decide(self, i, book, md):
        free = self.max_pos - len(book.positions) - len(book.pending)
        if free <= 0:
            return []
        out = []
        for score, sym in sorted(self.signals.get(i, []), reverse=True):
            if len(out) >= min(self.max_new, free) or sym in book.positions:
                continue
            out.append(Order(sym, cash_frac=1.0 / (free - len(out)), tp=self.tp, sl=self.sl, max_days=self.max_days,
                             reason="second-wave setup"))
        return out


SECOND_WAVE = SecondWave("second_wave", "Second wave (surge, pullback, volume building)")
SECOND_WAVE_NOVOL = SecondWave("second_wave_novol", "Second wave without the volume rule (control)", use_volume=False)
