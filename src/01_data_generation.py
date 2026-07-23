"""
01_data_generation.py
─────────────────────
Generates ≥ 100,000 synthetic records for three BODS-like catalogues:
  • timetables   – scheduled journey data
  • disruptions  – service disruption events
  • locations    – GPS pings per journey

Why Pandas here?
    Small, in-memory generation step; Faker is a pure-Python library
    with no Spark integration. PySpark is used in all downstream steps.

ST5011CEM | Night Bus Service Reliability Prediction
"""

import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from config.config import DATA_DIR, SYNTHETIC_RECORDS, RANDOM_SEED

fake = Faker("en_GB")
rng  = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

os.makedirs(DATA_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────
OPERATORS = ["NightLine Ltd", "MidCity Transit", "UrbanNight Express",
             "Capital Night Buses", "StarRoute Services"]
NIGHT_ROUTES = [f"N{i}" for i in range(1, 31)]          # N1 – N30
STOPS = [f"STOP_{i:04d}" for i in range(1, 201)]
START_DATE = datetime(2024, 10, 1)
END_DATE   = datetime(2025,  1, 1)   # 3-month window (Oct–Dec 2024)
NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))   # 22:00–05:59
WEATHER = ["Clear", "Rain", "Fog", "Snow", "Windy"]
WEATHER_PROBS = [0.55, 0.25, 0.10, 0.05, 0.05]


def random_night_datetime(start=START_DATE, end=END_DATE):
    """Return a random datetime on a random night within the 3-month window."""
    delta = (end - start).days
    day_offset = rng.integers(0, delta)
    base = start + timedelta(days=int(day_offset))
    hour = random.choice(NIGHT_HOURS)
    minute = rng.integers(0, 60)
    return base.replace(hour=hour, minute=int(minute), second=0, microsecond=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TIMETABLES  (80 000 rows)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating timetables …")
n_timetable = 80_000
scheduled_dt = [random_night_datetime() for _ in range(n_timetable)]
actual_offsets = rng.normal(loc=1.8, scale=4.5, size=n_timetable)   # mean 1.8 min late

timetables = pd.DataFrame({
    "trip_id":           [f"T{i:07d}" for i in range(n_timetable)],
    "service_code":      rng.choice(NIGHT_ROUTES, n_timetable),
    "operator":          rng.choice(OPERATORS, n_timetable),
    "origin_stop":       rng.choice(STOPS, n_timetable),
    "destination_stop":  rng.choice(STOPS, n_timetable),
    "scheduled_depart":  scheduled_dt,
    "actual_depart":     [
        scheduled_dt[i] + timedelta(minutes=float(actual_offsets[i]))
        for i in range(n_timetable)
    ],
    "scheduled_duration_min": rng.integers(20, 90, n_timetable),
    "actual_duration_min":    rng.integers(18, 110, n_timetable),
    "weather":           rng.choice(WEATHER, n_timetable, p=WEATHER_PROBS),
    "day_of_week":       [scheduled_dt[i].strftime("%A") for i in range(n_timetable)],
    "hour_of_night":     [scheduled_dt[i].hour for i in range(n_timetable)],
    "vehicle_age_years": rng.integers(1, 15, n_timetable),
    "passenger_count":   rng.integers(0, 75, n_timetable),
    "capacity":          [75] * n_timetable,
    "driver_experience_yrs": rng.integers(1, 30, n_timetable),
})

timetables["delay_minutes"] = (
    timetables["actual_depart"] - timetables["scheduled_depart"]
).dt.total_seconds() / 60

timetables["is_delayed"] = (
    timetables["delay_minutes"].abs() > 2
).astype(int)   # urban ± 2 min threshold

timetables["service_reliability_flag"] = (
    timetables["is_delayed"] == 0
).astype(int)

timetables["load_factor"] = (
    timetables["passenger_count"] / timetables["capacity"]
).clip(0, 1)

path_tt = os.path.join(DATA_DIR, "timetables.csv")
timetables.to_csv(path_tt, index=False)
print(f"  ✓ timetables: {len(timetables):,} rows → {path_tt}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DISRUPTIONS  (15 000 rows)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating disruptions …")
n_dis = 15_000
DISRUPTION_TYPES = ["Road Closure", "Accident", "Flooding",
                    "Police Incident", "Vehicle Breakdown", "Special Event"]

disruptions = pd.DataFrame({
    "disruption_id":   [f"D{i:06d}" for i in range(n_dis)],
    "service_code":    rng.choice(NIGHT_ROUTES, n_dis),
    "operator":        rng.choice(OPERATORS, n_dis),
    "disruption_type": rng.choice(DISRUPTION_TYPES, n_dis),
    "start_time":      [random_night_datetime() for _ in range(n_dis)],
    "duration_min":    rng.integers(5, 180, n_dis),
    "severity":        rng.choice(["Low", "Medium", "High"], n_dis,
                                  p=[0.50, 0.35, 0.15]),
    "affected_stops":  rng.integers(1, 20, n_dis),
    "resolved":        rng.choice([True, False], n_dis, p=[0.85, 0.15]),
})

path_dis = os.path.join(DATA_DIR, "disruptions.csv")
disruptions.to_csv(path_dis, index=False)
print(f"  ✓ disruptions: {len(disruptions):,} rows → {path_dis}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GPS LOCATIONS  (25 000 rows)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating GPS locations …")
n_loc = 25_000
sample_trips = timetables["trip_id"].sample(n_loc, replace=True,
                                             random_state=RANDOM_SEED).values

locations = pd.DataFrame({
    "location_id":   [f"L{i:07d}" for i in range(n_loc)],
    "trip_id":       sample_trips,
    "timestamp":     [random_night_datetime() for _ in range(n_loc)],
    "latitude":      rng.uniform(51.3, 51.7, n_loc),   # Greater London bounding box
    "longitude":     rng.uniform(-0.5,  0.2, n_loc),
    "speed_kmh":     rng.uniform(0, 60, n_loc),
    "heading_deg":   rng.uniform(0, 360, n_loc),
    "signal_quality": rng.choice(["Good", "Fair", "Poor"], n_loc,
                                  p=[0.70, 0.20, 0.10]),
})

path_loc = os.path.join(DATA_DIR, "locations.csv")
locations.to_csv(path_loc, index=False)
print(f"  ✓ locations: {len(locations):,} rows → {path_loc}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
total = len(timetables) + len(disruptions) + len(locations)
print(f"\n{'─'*55}")
print(f"  TOTAL records generated: {total:,}  (threshold: ≥ 100,000)")
print(f"  Date window: {START_DATE.date()} → {END_DATE.date()} (3 months)")
print(f"{'─'*55}")
