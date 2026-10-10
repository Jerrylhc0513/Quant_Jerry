"""
stealth.py - quiet accumulation in small/mid caps: volume rising steadily while the price moves sideways.
Buy when the walk-forward model rates the setup in its top 10%, then hold until +50% / -20% / 126 days.
Details of the pool, features and model: event.py (SETUPS["stealth"]).
"""
from event_strategy import EventStrategy

STEALTH = EventStrategy("stealth", "Stealth accumulation (months)", "stealth",
                        "Small/mid caps with rising volume and a sideways price; +50% / -20% / 126 days")
