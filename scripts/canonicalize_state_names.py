"""
Fix a data-quality bug in state_year_panel.csv: different RBI Handbook tables
spelled some states inconsistently ("Andaman & Nicobar Islands" vs "...and...",
"Jammu & Kashmir" vs "Jammu and Kashmir"), which extract_rbi_handbook.py's
per-table matching treated as distinct states, producing separate rows with
disjoint columns filled for what should be one state-year observation.

This coalesces every alias down to one canonical name and merges rows sharing
(canonical name, year) by taking the first non-null value per column - safe
because the two rows never have overlapping populated columns for the same
variable (they came from different source tables to begin with).

Run this once; also fixed at the source in extract_rbi_handbook.py's
STATE_NAMES list for future re-extractions.
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")

CANONICAL_MAP = {
    "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Dadra and Nagar Haveli": "Dadra & Nagar Haveli",
    "Daman and Diu": "Daman & Diu",
}


def main():
    df = pd.read_csv(PANEL_PATH)
    before = df["State"].nunique()
    df["State"] = df["State"].replace(CANONICAL_MAP)
    after = df["State"].nunique()

    df = df.groupby(["State", "Year_Standardized"], as_index=False).first()
    # first() on a groupby with NaNs still picks the first row's NaN if the
    # first row lacks a value another row in the group has - use combine instead.
    df2 = pd.read_csv(PANEL_PATH)
    df2["State"] = df2["State"].replace(CANONICAL_MAP)

    def coalesce_group(g):
        return g.bfill().ffill().iloc[0]

    merged = df2.groupby(["State", "Year_Standardized"], as_index=False).apply(coalesce_group).reset_index(drop=True)

    print(f"States before: {before} -> after canonicalization: {after}")
    print(f"Rows before: {len(df2)} -> after coalescing duplicates: {len(merged)}")

    merged.to_csv(PANEL_PATH, index=False)
    print(f"Saved: {PANEL_PATH}")


if __name__ == "__main__":
    main()
