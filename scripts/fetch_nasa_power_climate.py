"""
Pull per-state annual mean temperature (and a rainfall cross-check) from the
NASA POWER climate API, using each state's polygon-vertex-averaged centroid
from data/raw/state_panel/state_boundaries/india_state.geojson.

This replaces the old single national-centroid Open-Meteo pull in
scripts/acquire_data_api.py with a genuinely per-state series, and needs no
manual download or registration (NASA POWER is a public, unauthenticated API).

Caveats (documented, not silently fixed):
- The boundary file predates the 2014 Telangana/Andhra Pradesh split and the
  2019 Jammu & Kashmir/Ladakh split, so those states reuse their pre-split
  parent's centroid - a coarse approximation until updated boundaries are
  sourced.
- A polygon-vertex average is not a true area-weighted centroid, but is a
  reasonable single representative point per state for a point-based climate
  API.
"""
import os
import json
import time
import requests
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _panel_io import upsert_into_panel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "raw", "state_panel", "state_boundaries", "india_state.geojson")
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")
OUT_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "state_panel", "imd_climate", "nasa_power_state_climate.csv")

START_YEAR, END_YEAR = 1994, 2024

# GADM-era name -> RBI Handbook canonical name used in state_year_panel.csv
NAME_MAP = {
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand",
    "Andaman and Nicobar": "Andaman & Nicobar Islands",
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Dadra and Nagar Haveli": "Dadra & Nagar Haveli",
    "Daman and Diu": "Daman & Diu",
}
# States absent from the pre-2014/2019 boundary file: reuse the parent state's
# centroid as a documented approximation until updated boundaries are sourced.
MANUAL_CENTROIDS = {
    "Telangana": (78.9, 17.9),  # ~Hyderabad; AP's own centroid otherwise double-serves both
}


def flatten(coords):
    if isinstance(coords[0][0], (int, float)):
        return [coords]
    pts = []
    for c in coords:
        pts.extend(flatten(c))
    return pts


def compute_centroids():
    d = json.load(open(GEOJSON_PATH, encoding="utf-8"))
    centroids = {}
    for f in d["features"]:
        name = f["properties"]["NAME_1"]
        rings = flatten(f["geometry"]["coordinates"])
        xs = [pt[0] for ring in rings for pt in ring]
        ys = [pt[1] for ring in rings for pt in ring]
        canonical = NAME_MAP.get(name, name)
        centroids[canonical] = (sum(xs) / len(xs), sum(ys) / len(ys))
    centroids.update(MANUAL_CENTROIDS)
    return centroids


def fetch_power_point(lon, lat, state_name):
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
        f"&start={START_YEAR}&end={END_YEAR}&format=JSON"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]
    rows = []
    for ym, temp in data["T2M"].items():
        year, month = int(ym[:4]), ym[4:6]
        if month == "13":  # annual summary key some months include; skip, we aggregate ourselves
            continue
        precip = data["PRECTOTCORR"].get(ym)
        rows.append({"State": state_name, "Year_Standardized": year, "T2M": temp, "PRECTOTCORR_mm_day": precip})
    df = pd.DataFrame(rows)
    annual = df.groupby(["State", "Year_Standardized"]).agg(
        Temperature_C=("T2M", "mean"),
        NASA_Rainfall_mm=("PRECTOTCORR_mm_day", lambda s: s.mean() * 30.44 * 12),  # rough mm/day -> mm/year
    ).reset_index()
    return annual


def main():
    os.makedirs(os.path.dirname(OUT_RAW_PATH), exist_ok=True)
    centroids = compute_centroids()
    all_annual = []
    for i, (state, (lon, lat)) in enumerate(centroids.items()):
        print(f"[{i+1}/{len(centroids)}] Fetching {state} ({lat:.2f}, {lon:.2f})...")
        try:
            annual = fetch_power_point(lon, lat, state)
            all_annual.append(annual)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(0.5)  # be polite to the public API

    combined = pd.concat(all_annual, ignore_index=True)
    combined.to_csv(OUT_RAW_PATH, index=False)
    print(f"\nSaved raw climate pull: {OUT_RAW_PATH} ({combined.shape})")

    panel = pd.read_csv(PANEL_PATH)
    merged = upsert_into_panel(panel, combined)
    merged.to_csv(PANEL_PATH, index=False)
    print(f"Merged into {PANEL_PATH} | Temperature populated for {merged['Temperature_C'].notna().mean():.1%} of rows")


if __name__ == "__main__":
    main()
