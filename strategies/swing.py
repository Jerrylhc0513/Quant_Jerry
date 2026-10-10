"""
swing.py - hot stocks right after a price or volume burst; hold for days: +8% / -5% / 10 days.
Details of the pool, features and model: event.py (SETUPS["swing"]).
"""
from event_strategy import EventStrategy

SWING = EventStrategy("swing", "Hot-stock swing (days)", "swing",
                      "Liquid stocks after a 10% / 5-day jump or doubled volume; +8% / -5% / 10 days")
