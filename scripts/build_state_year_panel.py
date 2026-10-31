"""
Consolidate the long-format RBI Handbook table extracts into a single
state x year panel (data/processed/state_panel/state_year_panel.csv).

Fiscal-year labels ("2013-14") are standardized to their starting calendar
year (2013) to match the convention used elsewhere in this project
(Year_Standardized). Table 102 (subdivision-level rainfall) is intentionally
excluded here: IMD meteorological subdivisions don't map 1:1 onto
administrative states (e.g. "Assam & Meghalaya" spans two states, Uttar
Pradesh is split into "East"/"West"), so it needs an explicit, documented
subdivision -> state crosswalk before merging - not done automatically.
"""
import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "rbi_tables", "_all_tables_long.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")

EXCLUDED_TABLE_IDS = {102}  # subdivision-level, needs a separate crosswalk step


def standardize_year(fiscal_year):
    """'2013-14' -> 2013; '2013' -> 2013."""
    m = re.match(r"(\d{4})", str(fiscal_year))
    return int(m.group(1)) if m else None


def main():
    df = pd.read_csv(IN_PATH)
    df = df[~df["Table_ID"].isin(EXCLUDED_TABLE_IDS)].copy()
    df["Year_Standardized"] = df["Fiscal_Year"].apply(standardize_year)

    # A given (State, Year, Variable) should be unique; if a state/year appears
    # in more than one source table under the same variable name, keep the first.
    dup_key = ["State", "Year_Standardized", "Variable"]
    spread = df.groupby(dup_key)["Value"].agg(["nunique", "count"])
    conflicts = spread[(spread["count"] > 1) & (spread["nunique"] > 1)]
    print(f"Duplicate (State, Year, Variable) groups with CONFLICTING values: {len(conflicts)} "
          f"(first value kept; see below)")
    if len(conflicts):
        print(conflicts.head(20).to_string())
    df = df.drop_duplicates(subset=dup_key, keep="first")

    panel = df.pivot_table(index=["State", "Year_Standardized"], columns="Variable", values="Value", aggfunc="first")
    panel = panel.reset_index().sort_values(["State", "Year_Standardized"])

    n_states = panel["State"].nunique()
    n_years = panel["Year_Standardized"].nunique()
    print(f"Panel shape: {panel.shape} | {n_states} states x {panel['Year_Standardized'].min()}-{panel['Year_Standardized'].max()}")
    print(f"Variables: {[c for c in panel.columns if c not in ('State', 'Year_Standardized')]}")

    non_null_frac = panel.drop(columns=["State", "Year_Standardized"]).notna().mean().sort_values()
    print("\nCompleteness by variable (fraction of state-years populated):")
    print(non_null_frac.to_string())

    # Re-running must not wipe the columns later steps merge in (rainfall, temperature, census, livestock, electricity):
    # new RBI values override old ones where present, everything else already in the file is kept.
    if os.path.exists(OUT_PATH):
        keys = ["State", "Year_Standardized"]
        existing = pd.read_csv(OUT_PATH)
        panel = panel.set_index(keys).combine_first(existing.set_index(keys)).reset_index().sort_values(keys)
    panel.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
