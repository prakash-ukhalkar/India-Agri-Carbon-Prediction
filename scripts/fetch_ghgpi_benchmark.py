"""
Download and tidy the GHG Platform India (GHGPI) state-level AFOLU estimates, used as an INDEPENDENT-METHOD
benchmark for the bottom-up panel (and as an alternative ML target).

Source: GHG Platform India, "AFOLU Emissions 2005 to 2018" (Vasudha Foundation), version 4.0, 15 Sept 2022,
        https://ghgplatform-india.org/afolu-sector/   (CO2e uses IPCC AR2 GWPs, as in India's BUR).
Usage policy: the page has a "Usage Policy" section whose text could not be retrieved; no licence is stated.
        The raw workbook is therefore downloaded on demand and NOT redistributed here; only the tidy derived
        numbers used for validation are written. CHECK THE PLATFORM'S USAGE POLICY BEFORE PUBLISHING THE CSV.
Caveat: GHGPI uses the same public activity data (Livestock Census, DES state area/production, fertiliser
        statistics), so agreement validates implementation, not the underlying activity data or emission factors.

Output: data/processed/state_panel/ghgpi_state_agri_2005_2018.csv
        columns: State, Year_Standardized, <category>_ktCO2e ..., GHGPI_agri_gross_ktCO2e
"""
import os
import urllib.request
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://ghgplatform-india.org/wp-content/uploads/2022/09/GHGPI-AFOLU-Estimates-2005-to-2018-by-Vasudha-Foundation.xlsx"
RAW = os.path.join(BASE, "data", "raw", "benchmark_ghgpi", "GHGPI-AFOLU-Estimates-2005-to-2018.xlsx")
OUT = os.path.join(BASE, "data", "processed", "state_panel", "ghgpi_state_agri_2005_2018.csv")

# GHGPI state spellings -> this project's canonical names
STATE_MAP = {"Andaman and Nicobar Island": "Andaman & Nicobar Islands", "Dadra and Nagar Haveli": "Dadra & Nagar Haveli",
             "Daman and Diu": "Daman & Diu", "Jammu and Kashmir": "Jammu & Kashmir"}
CATEGORY = {"Enteric Fermentation": "enteric", "Manure Management": "manure", "Rice Cultivation": "rice",
            "Agriculture Soils": "soils", "Biomass Burning in Cropland": "cropland_burning"}


def main():
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        print("downloading", URL)
        urllib.request.urlretrieve(URL, RAW)
    d = pd.read_excel(RAW, sheet_name="GHG Estimation")
    d = d[(d["Gas"] == "CO2e (t) GWP-AR2") & (d["Emission / Removal / Bunker"] == "Emissions")
          & d["Level 3"].isin(CATEGORY)].copy()
    years = [c for c in d.columns if isinstance(c, (int, float)) and 2005 <= c <= 2018]
    d["cat"] = d["Level 3"].map(CATEGORY)
    d["State"] = d["State"].replace(STATE_MAP)
    long = d.melt(id_vars=["State", "cat"], value_vars=years, var_name="Year_Standardized", value_name="t")
    long["Year_Standardized"] = long["Year_Standardized"].astype(int)
    wide = (long.groupby(["State", "Year_Standardized", "cat"])["t"].sum().div(1000.0)  # t -> kt
            .unstack("cat").add_suffix("_ktCO2e").reset_index())
    cols = [c for c in wide.columns if c.endswith("_ktCO2e")]
    wide["GHGPI_agri_gross_ktCO2e"] = wide[cols].sum(axis=1)
    wide.to_csv(OUT, index=False)
    nat = wide[wide.Year_Standardized == 2018][cols].sum() / 1000
    print(f"saved {OUT} | {wide.State.nunique()} states x {wide.Year_Standardized.min()}-{wide.Year_Standardized.max()}")
    print("national 2018 (Mt CO2e, AR2):", nat.round(2).to_dict(), "| total", round(nat.sum(), 2),
          "(published: 351.55 incl. forest burning 1.51 -> 350.04 agricultural)")


if __name__ == "__main__":
    main()
