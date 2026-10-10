"""
event_strategy.py - adapter that runs an event.py setup inside the engine.

prepare(): candidate pool, features and walk-forward probabilities for every day (each year's model only
learns from trades whose outcome was known before that year). decide(): buy the best-rated new candidates
while slots are free, with the setup's profit line, stop line and holding limit attached.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

import event as EV
from engine import Order, Strategy

log = logging.getLogger("event_strategy")


class EventStrategy(Strategy):
    def __init__(self, name, title, setup: str, description="", **overrides):
        self.name, self.title, self.description = name, title, description
        self.setup = replace(EV.SETUPS[setup], **overrides) if overrides else EV.SETUPS[setup]
        self.cost_bps = self.setup.cost_bps

    def prepare(self, md):
        s = self.setup
        pn = EV.panels_from(md, s)
        f = EV.features(pn)
        pool = EV.candidate_pool(s, pn, f)
        ret, days, res = EV.outcomes(pn, s.tp, s.sl, s.max_days)
        ds = EV.build_dataset(pool, f, ret, days, res)
        del f
        prob, thr = EV.walk_forward(ds, md.dates, s.max_days)
        year = md.dates.year.to_numpy()[ds.ti]
        ok = np.isfinite(prob) & (prob >= np.array([thr.get(y, np.inf) for y in year]))
        syms = pn["C"].columns
        self.picks: dict[int, list[tuple[float, str]]] = {}
        for t, n, p in zip(ds.ti[ok], ds.ni[ok], prob[ok]):
            self.picks.setdefault(int(t), []).append((float(p), syms[n]))
        tested = np.flatnonzero(np.isfinite(prob))
        if not len(tested):
            raise SystemExit(f"{self.name}: too few past setups to train a model ({len(ds.ti)} candidates)")
        self._first = int(ds.ti[tested].min()) if len(tested) else len(md.dates) - 1

    def first_day(self, md):
        return self._first

    def decide(self, i, book, md):
        s = self.setup
        free = s.max_pos - len(book.positions) - len(book.pending)
        if free <= 0:
            return []
        out = []
        for p, sym in sorted(self.picks.get(i, []), reverse=True):
            if len(out) >= min(s.max_new, free) or sym in book.positions:
                continue
            out.append(Order(sym, cash_frac=1.0 / (free - len(out)), tp=s.tp, sl=s.sl, max_days=s.max_days,
                             reason=f"setup p={p:.2f}"))
        return out
