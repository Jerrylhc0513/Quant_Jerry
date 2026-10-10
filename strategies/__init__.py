"""
strategies/ - one idea per file. Every module-level instance of engine.Strategy in any file here is picked up
by run_strategies.py, so a file may hold one strategy or several variants. Set `enabled = False` on a strategy
to skip it. Files starting with "_" are ignored.
"""
from __future__ import annotations

import importlib
import pkgutil

from engine import Strategy


def discover(include_disabled: bool = False) -> dict[str, Strategy]:
    found: dict[str, Strategy] = {}
    for m in pkgutil.iter_modules(__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{m.name}")
        for obj in vars(mod).values():
            if isinstance(obj, Strategy) and (obj.enabled or include_disabled):
                if obj.name in found and found[obj.name] is not obj:
                    raise SystemExit(f"Two strategies are both named '{obj.name}'")
                found[obj.name] = obj
    return dict(sorted(found.items()))
