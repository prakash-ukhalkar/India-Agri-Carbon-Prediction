"""
Parse state-wise TOTAL livestock headcounts by species for the 19th Livestock
Census (2012) from its "All India Report" PDF, using the Rural+Urban combined
tables (e.g. "Table 14 (R+U) ... Total Cattle"). The report's per-species
detail tables break animals down by age/sex/use, but each combined (R+U)
table's LAST numeric column on each row is always that species' state total -
confirmed by manual inspection of every table used here.

State names in this PDF are rendered inconsistently (spaces sometimes
stripped, e.g. "ANDHRAPRADESH", and long names wrap onto a second line with
no numbers). Rather than fuzzy-match name strings, this relies on the fixed,
consistent S.No. ordering (1-35, same alphabetical order in every table in
this report) to assign each numeric row to its state - far more robust than
text matching against garbled names.

Merges into the panel as Year_Standardized=2012, alongside the existing 2019
Livestock Census data, so livestock now has 2 real time points instead of 1
(within-state linear interpolation between them can be added in a later
feature-engineering step, never fabricating cross-state values).
"""
import os
import re
import pdfplumber
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _panel_io import upsert_into_panel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "raw", "state_panel", "livestock_census", "19thLivestockCensus2012.pdf")
PANEL_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "state_year_panel.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "state_panel", "livestock_census_2012.csv")

YEAR = 2012

# Fixed alphabetical S.No. order used consistently across every table in this
# report (confirmed against Table 14, page 61). Telangana did not yet exist.
STATE_ORDER = [
    "Andaman & Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra & Nagar Haveli", "Daman & Diu", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand",
    "Karnataka", "Kerala", "Lakshadweep", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Delhi", "Odisha",
    "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
]

# page (0-indexed) -> (output column, which numeric value on the row to take)
# "last" = last number in the row; otherwise a 0-indexed position among the
# numbers actually present on that row.
SPECIES_TABLES = {
    61: ("Livestock_Cattle_Head", "last"),
    67: ("Livestock_Buffalo_Head", "last"),
    70: ("Livestock_Mithun_Head", "last"),
    73: ("Livestock_Yaks_Head", "last"),
    79: ("Livestock_Sheep_Head", "last"),
    82: ("Livestock_Goat_Head", "last"),
    88: ("Livestock_Pigs_Head", "last"),
    91: ("Livestock_Horse_and_Ponies_Head", "last"),
    97: ("Livestock_Camel_Head", "last"),
}
# Mules and Donkeys share one table with two totals to extract separately.
MULES_DONKEYS_PAGE = 94

ROW_RE = re.compile(r"^(\d{1,2})\s+.+?\s+((?:\d+\s+)+\d+)$")


def parse_species_page(page_text, value_pos):
    values_by_sno = {}
    for line in page_text.split("\n"):
        line = line.strip()
        m = ROW_RE.match(line)
        if not m:
            continue
        sno = int(m.group(1))
        if sno < 1 or sno > len(STATE_ORDER):
            continue
        numbers = m.group(2).split()
        idx = -1 if value_pos == "last" else value_pos
        values_by_sno[sno] = float(numbers[idx])
    return values_by_sno


def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        data = {}  # state -> {column: value}
        for page_idx, (col, pos) in SPECIES_TABLES.items():
            text = pdf.pages[page_idx].extract_text()
            values = parse_species_page(text, pos)
            for sno, val in values.items():
                state = STATE_ORDER[sno - 1]
                data.setdefault(state, {})[col] = val
            print(f"Page {page_idx} ({col}): {len(values)}/{len(STATE_ORDER)} states parsed")

        # Mules and Donkeys: 10 numbers per row, TotalMules = index 2, TotalDonkeys = index 9
        text = pdf.pages[MULES_DONKEYS_PAGE].extract_text()
        for line in text.split("\n"):
            line = line.strip()
            m = ROW_RE.match(line)
            if not m:
                continue
            sno = int(m.group(1))
            if sno < 1 or sno > len(STATE_ORDER):
                continue
            numbers = m.group(2).split()
            if len(numbers) < 10:
                continue
            state = STATE_ORDER[sno - 1]
            data.setdefault(state, {})["Livestock_Mules_Head"] = float(numbers[2])
            data.setdefault(state, {})["Livestock_Donkeys_Head"] = float(numbers[9])
        print(f"Page {MULES_DONKEYS_PAGE} (Mules & Donkeys): {sum(1 for s in data if 'Livestock_Mules_Head' in data[s])}/{len(STATE_ORDER)} states parsed")

    df = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "State"})
    df["Year_Standardized"] = YEAR
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} states -> {OUT_PATH}")
    print(df.to_string(index=False))

    panel = pd.read_csv(PANEL_PATH)
    merged = upsert_into_panel(panel, df)
    merged.to_csv(PANEL_PATH, index=False)
    n_matched = merged.loc[merged["Year_Standardized"] == YEAR, "Livestock_Cattle_Head"].notna().sum()
    print(f"\nMerged into panel: {n_matched} of {YEAR} rows now populated with livestock headcounts")


if __name__ == "__main__":
    main()
