"""
Map Table 102's IMD meteorological subdivisions onto administrative states and
produce a state x year rainfall series, merged into the RBI state_year_panel.

IMD subdivisions don't align 1:1 with states:
- Some states are covered by a single subdivision (Punjab, Kerala, ...).
- Some states are split across multiple subdivisions (UP -> East+West, Rajasthan
  -> East+West, Madhya Pradesh -> East+West, Gujarat -> Region+Saurashtra&Kutch,
  Maharashtra -> Konkan&Goa+Madhya Maharashtra+Marathwada+Vidarbha, Karnataka ->
  Coastal+North Interior+South Interior). These are combined with a simple
  unweighted mean across their subdivisions - a documented approximation
  (area-weighting would be more precise but requires subdivision shapefiles).
- Some subdivisions are shared by multiple states (Assam & Meghalaya; Naga/
  Mani/Mizo/Tripura; Haryana, Delhi & Chandigarh; Himalayan West Bengal &
  Sikkim). The same subdivision value is assigned to each state sharing it.
- West Bengal itself = Gangetic West Bengal (its own dedicated subdivision).
"""
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _panel_io import upsert_into_panel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAINFALL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "rbi_tables", "table_102_Subdivision_Annual_Rainfall.csv")
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")

# Merge the two spellings of the same subdivision that appear in different years.
SUBDIVISION_ALIASES = {
    "Naga, Mani, Mizo & Tripura": "Naga Mani Mizo Tripura",
}

# subdivision -> list of states it represents (single-item = 1:1, multi-item = shared)
SUBDIVISION_TO_STATES = {
    "Andaman & Nicobar Islands": ["Andaman & Nicobar Islands"],
    "Arunachal Pradesh": ["Arunachal Pradesh"],
    "Assam & Meghalaya": ["Assam", "Meghalaya"],
    "Naga Mani Mizo Tripura": ["Nagaland", "Manipur", "Mizoram", "Tripura"],
    "Himalayan West Bengal & Sikkim": ["Sikkim"],
    "Gangetic West Bengal": ["West Bengal"],
    "Odisha": ["Odisha"],
    "Jharkhand": ["Jharkhand"],
    "Bihar": ["Bihar"],
    "East Uttar Pradesh": ["Uttar Pradesh"],
    "West Uttar Pradesh": ["Uttar Pradesh"],
    "Uttarakhand": ["Uttarakhand"],
    "Haryana, Delhi & Chandigarh": ["Haryana", "Delhi", "Chandigarh"],
    "Punjab": ["Punjab"],
    "Himachal Pradesh": ["Himachal Pradesh"],
    "Jammu & Kashmir": ["Jammu & Kashmir"],
    "West Rajasthan": ["Rajasthan"],
    "East Rajasthan": ["Rajasthan"],
    "West Madhya Pradesh": ["Madhya Pradesh"],
    "East Madhya Pradesh": ["Madhya Pradesh"],
    "Gujarat Region": ["Gujarat"],
    "Saurashtra & Kutch": ["Gujarat"],
    "Konkan & Goa": ["Maharashtra", "Goa"],
    "Madhya Maharashtra": ["Maharashtra"],
    "Marathwada": ["Maharashtra"],
    "Vidarbha": ["Maharashtra"],
    "Chhattisgarh": ["Chhattisgarh"],
    "Coastal Andhra Pradesh": ["Andhra Pradesh"],
    "Telangana": ["Telangana"],
    "Rayalseema": ["Andhra Pradesh"],
    "Tamil Nadu": ["Tamil Nadu"],
    "Coastal Karnataka": ["Karnataka"],
    "North Interior Karnataka": ["Karnataka"],
    "South Interior Karnataka": ["Karnataka"],
    "Kerala": ["Kerala"],
    "Lakshadweep": ["Lakshadweep"],
}


def main():
    df = pd.read_csv(RAINFALL_PATH)
    df["Subdivision"] = df["Subdivision"].replace(SUBDIVISION_ALIASES)
    df["Year_Standardized"] = df["Fiscal_Year"].astype(int)

    rows = []
    for _, r in df.iterrows():
        states = SUBDIVISION_TO_STATES.get(r["Subdivision"])
        if states is None:
            continue
        for state in states:
            rows.append({"State": state, "Year_Standardized": r["Year_Standardized"], "Value": r["Value"]})

    long_df = pd.DataFrame(rows)
    # States split across multiple subdivisions (e.g. Uttar Pradesh, Rajasthan,
    # Madhya Pradesh, Gujarat, Maharashtra, Karnataka) get an unweighted mean.
    state_rainfall = long_df.groupby(["State", "Year_Standardized"])["Value"].mean().reset_index()
    state_rainfall.rename(columns={"Value": "Annual_Rainfall_mm"}, inplace=True)

    print(f"State-level rainfall: {state_rainfall.shape} | {state_rainfall['State'].nunique()} states x "
          f"{state_rainfall['Year_Standardized'].min()}-{state_rainfall['Year_Standardized'].max()}")

    panel = pd.read_csv(PANEL_PATH)
    merged = upsert_into_panel(panel, state_rainfall)
    merged.to_csv(PANEL_PATH, index=False)
    coverage = merged["Annual_Rainfall_mm"].notna().mean()
    print(f"Merged into {PANEL_PATH} | rainfall populated for {coverage:.1%} of panel rows")


if __name__ == "__main__":
    main()
