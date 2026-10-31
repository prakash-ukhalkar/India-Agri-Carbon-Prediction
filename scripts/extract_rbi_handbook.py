"""
Extract state-wise time series tables from the RBI Handbook of Statistics on
Indian States (2024-25) PDF into tidy long-format CSVs.

Each target table is rendered as plain text (no gridlines), typically split
across two pages: a first page with years ...-2013-14 and a second
"(Concld.)" page with 2014-15 onward. This parser matches each line against a
canonical state/UT name list, pulls the trailing numeric columns, and maps
them to the year header parsed from that page.

Usage: run from repo root with the project venv:
    venv_agri_carbon/Scripts/python.exe scripts/extract_rbi_handbook.py
"""
import os
import re
import pdfplumber
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "raw", "state_panel", "rbi_nsdp", "Handbook_2024-25.pdf")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "state_panel", "rbi_tables")
os.makedirs(OUT_DIR, exist_ok=True)

# Canonical state/UT names, longest-first so multi-word names match before
# their single-word prefixes (e.g. "Andaman and Nicobar Islands" before "Andaman").
# Canonical name -> spelling variants seen across different Handbook tables.
# Every variant below resolves to the FIRST (canonical) name so a state never
# ends up split across two differently-spelled rows in the panel.
STATE_NAME_ALIASES = {
    "Andaman & Nicobar Islands": ["Andaman and Nicobar Islands"],
    "Jammu & Kashmir": ["Jammu and Kashmir"],
    "Dadra & Nagar Haveli": ["Dadra and Nagar Haveli", "Dadra & Nagar Haveli and Daman & Diu"],
    "Daman & Diu": ["Daman and Diu"],
    "Uttarakhand": ["Uttaranchal"],
    "Odisha": ["Orissa"],
    "Delhi": ["NCT of Delhi"],
}
_CANONICAL_NAMES = [
    "Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu", "Puducherry",
    "Andaman & Nicobar Islands", "Lakshadweep",
    "Gujarat", "Madhya Pradesh", "Chhattisgarh", "Maharashtra", "Rajasthan", "Goa",
    "Daman & Diu", "Dadra & Nagar Haveli",
    "Haryana", "Punjab", "Uttar Pradesh", "Uttarakhand",
    "Himachal Pradesh", "Jammu & Kashmir", "Ladakh", "Delhi", "Chandigarh",
    "Bihar", "Jharkhand", "Odisha", "West Bengal",
    "Assam", "Tripura", "Manipur", "Meghalaya", "Nagaland", "Arunachal Pradesh",
    "Mizoram", "Sikkim", "All India",
]
# Resolve every alias to its canonical name; parse_page() matches on this full
# list, and extract_table() maps whichever variant matched back to canonical.
ALIAS_TO_CANONICAL = {name: name for name in _CANONICAL_NAMES}
for canonical, variants in STATE_NAME_ALIASES.items():
    for v in variants:
        ALIAS_TO_CANONICAL[v] = canonical

STATE_NAMES = sorted(ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)

NOT_APPLICABLE = {"-", ".", "..", "—", "–", "NA", "N.A."}


def parse_number(tok):
    tok = tok.strip()
    # The Handbook flags estimated/provisional values with a footnote marker after the number (e.g. "10114*"). Strip it:
    # the earlier version dropped such cells silently (127 cells in land-use, livestock-product and groundwater tables).
    tok = tok.rstrip("*#^+")
    if tok in NOT_APPLICABLE or tok == "":
        return None
    tok = tok.replace(",", "")
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()")
    try:
        val = float(tok)
        return -val if neg else val
    except ValueError:
        return None


def parse_year_header(text):
    """Extract the sequence of year labels from a page's header line.
    Tries fiscal-year ranges (2013-14) first, falls back to plain 4-digit years
    (2012 2013 ...) for tables that report calendar years. Scans every line
    within the first ~8 lines and keeps the one with the most year tokens,
    since lines like "Base: 2011-12" would otherwise be mistaken for the header."""
    best = []
    for line in text.split("\n")[:8]:
        years = re.findall(r"(\d{4}-\d{2,4})", line)
        if not years:
            years = re.findall(r"\b(\d{4})\b", line)
        if len(years) > len(best):
            best = years
    seen = []
    for y in best:
        if y not in seen:
            seen.append(y)
    return seen


MARKERS = "*#^+"


def _numeric_tokens(rest):
    """Tokens of a data row with footnote markers removed so they can never occupy (and shift) a value slot."""
    toks = []
    for t in rest.split():
        if t.strip(MARKERS) == "" and t not in NOT_APPLICABLE:
            continue            # a bare marker such as '#'
        toks.append(t)
    return toks


def parse_page(text, name_list=None):
    """Return list of (name, [values]) tuples found in a page's text, matching
    against name_list (defaults to STATE_NAMES). Tolerates footnote markers
    ('*', '#') directly after the name, and names that wrap onto neighbouring lines
    (RBI wraps long names: 'Andaman & Nicobar' / numbers / 'Islands').
    v2 (2026-10-31): previously a '#' after the name consumed a value slot, shifting every
    value of that row one year (Andhra Pradesh in the livestock-product tables), and wrapped
    names were skipped."""
    if name_list is None:
        name_list = STATE_NAMES
    rows = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        matched_name = None
        rest = None
        for name in name_list:
            m = re.match(re.escape(name) + r"[*#^+]*\s*", line)
            if m:
                matched_name = name
                rest = line[m.end():].strip()
                break
        if matched_name is None:
            # wrapped name: this line is a proper prefix of a known name, the numbers follow on the next line
            alpha = re.sub(r"[*#^+]+$", "", line).strip()
            if alpha and not re.search(r"\d", alpha) and len(alpha) >= 8 and i + 1 < len(lines) and re.search(r"\d", lines[i + 1]):
                for name in name_list:
                    if name.startswith(alpha) and name != alpha:
                        matched_name, rest = name, lines[i + 1]
                        i += 1
                        if i + 1 < len(lines) and name.endswith(lines[i + 1].strip()):
                            i += 1          # swallow the wrapped remainder ("Islands")
                        break
        if matched_name is None:
            i += 1
            continue
        # If the name matched but nothing numeric follows on this line, the
        # values are on the next line (RBI wraps long subdivision names).
        if not rest and i + 1 < len(lines) and re.search(r"\d", lines[i + 1]):
            rest = lines[i + 1]
            i += 1
        values = [parse_number(t) for t in _numeric_tokens(rest)]
        if any(v is not None for v in values):
            rows.append((matched_name, values))
        i += 1
    return rows


# IMD meteorological subdivisions used by Table 102 (rainfall) - NOT administrative
# states. Longest-first so multi-word names match before shorter substrings.
SUBDIVISION_NAMES = sorted([
    "Andaman & Nicobar Islands", "Arunachal Pradesh", "Assam & Meghalaya",
    "Naga Mani Mizo Tripura", "Naga, Mani, Mizo & Tripura",
    "Himalayan West Bengal & Sikkim", "Gangetic West Bengal", "Odisha",
    "Jharkhand", "Bihar", "East Uttar Pradesh", "West Uttar Pradesh",
    "Uttarakhand", "Haryana, Delhi & Chandigarh", "Punjab", "Himachal Pradesh",
    "Jammu & Kashmir", "West Rajasthan", "East Rajasthan", "West Madhya Pradesh",
    "East Madhya Pradesh", "Gujarat Region", "Saurashtra & Kutch",
    "Konkan & Goa", "Madhya Maharashtra", "Marathwada", "Vidarbha",
    "Chhattisgarh", "Coastal Andhra Pradesh", "Telangana", "Rayalseema",
    "Tamil Nadu", "Coastal Karnataka", "North Interior Karnataka",
    "South Interior Karnataka", "Kerala", "Lakshadweep",
], key=len, reverse=True)


def extract_table(pdf, page_indices, table_id, table_name, name_list=None, entity_col="State"):
    """Extract one table spanning possibly multiple pages into a long DataFrame."""
    records = {}  # name -> {year: value}
    for pi in page_indices:
        text = pdf.pages[pi].extract_text() or ""
        years = parse_year_header(text)
        if not years:
            continue
        for name, values in parse_page(text, name_list=name_list):
            if name_list is None or name_list is STATE_NAMES:
                name = ALIAS_TO_CANONICAL.get(name, name)
            d = records.setdefault(name, {})
            for y, v in zip(years, values):
                if v is not None:
                    d[y] = v

    long_rows = []
    for name, yearvals in records.items():
        for year, value in yearvals.items():
            long_rows.append({
                entity_col: name,
                "Fiscal_Year": year,
                "Table_ID": table_id,
                "Variable": table_name,
                "Value": value,
            })
    df = pd.DataFrame(long_rows)
    return df


def extract_table_107(pdf, page_indices, table_id="107"):
    """Table 107 (groundwater) has 3 metrics x 3 assessment years (2017/2020/2022)
    per row instead of a simple year sequence - parsed with a fixed layout."""
    metrics = ["Extractable_Ground_Water_Resource_bcm", "Annual_Ground_Water_Extraction_bcm", "Stage_of_Extraction_pct"]
    assess_years = ["2017", "2020", "2022"]
    long_rows = []
    for pi in page_indices:
        text = pdf.pages[pi].extract_text() or ""
        for name, values in parse_page(text, name_list=STATE_NAMES):
            name = ALIAS_TO_CANONICAL.get(name, name)
            if len(values) < 9:
                continue
            values = values[:9]
            for j, year in enumerate(assess_years):
                for k, metric in enumerate(metrics):
                    v = values[j * 3 + k]
                    if v is not None:
                        long_rows.append({
                            "State": name,
                            "Fiscal_Year": year,
                            "Table_ID": table_id,
                            "Variable": metric,
                            "Value": v,
                        })
    return pd.DataFrame(long_rows)


# (table_id, variable_name, 0-indexed pdf page numbers)
TABLE_DEFS = [
    (29, "GSVA_Agriculture_Current", [96, 97]),
    (30, "GSVA_Agriculture_Constant", [98, 99]),
    (31, "NSVA_Agriculture_Current", [100, 101]),
    (32, "NSVA_Agriculture_Constant", [102, 103]),
    (53, "Gross_Sown_Area", [146, 147]),
    (54, "Net_Sown_Area", [148, 149]),
    (55, "Gross_Irrigated_Area", [150, 151]),
    (56, "Net_Irrigated_Area", [152, 153]),
    (57, "Cropping_Intensity", [154, 155]),
    (58, "Rice_Production", [156, 157]),
    (59, "Wheat_Production", [158, 159]),
    (60, "Coarse_Cereals_Production", [160, 161]),
    (61, "Pulses_Production", [162, 163]),
    (62, "Total_Foodgrains_Production", [164, 165]),
    (63, "Oilseeds_Production", [166, 167]),
    (64, "Cotton_Production", [168, 169]),
    (65, "Sugarcane_Production", [170, 171]),
    (66, "Jute_Mesta_Production", [172, 173]),
    (67, "Rice_Area", [174, 175]),
    (68, "Wheat_Area", [176, 177]),
    (69, "Coarse_Cereals_Area", [178, 179]),
    (70, "Pulses_Area", [180, 181]),
    (71, "Total_Foodgrains_Area", [182, 183]),
    (76, "Oilseeds_Area", [192, 193]),
    (77, "Cotton_Area", [194, 195]),
    (78, "Sugarcane_Area", [196, 197]),
    (79, "Jute_Mesta_Area", [198, 199]),
    (80, "Rice_Yield", [200, 201]),
    (81, "Wheat_Yield", [202, 203]),
    (82, "Coarse_Cereals_Yield", [204, 205]),
    (83, "Pulses_Yield", [206, 207]),
    (84, "Total_Foodgrains_Yield", [208, 209]),
    (85, "Oilseeds_Yield", [210, 211]),
    (86, "Cotton_Yield", [212, 213]),
    (87, "Sugarcane_Yield", [214, 215]),
    (88, "Jute_Mesta_Yield", [216, 217]),
    (89, "Fertiliser_Consumption_NPK", [218]),
    (90, "Meat_Production", [219, 220]),
    (91, "Milk_Production", [221, 222]),
    (92, "Egg_Production", [223, 224]),
    (93, "Fish_Production", [225, 226]),
    (94, "Wool_Production", [227, 228]),
    (102, "Subdivision_Annual_Rainfall", [247, 248, 249]),
    (103, "Cold_Wave_Days", [250]),
    (104, "Heat_Wave_Days", [251]),
    (107, "Groundwater_Extraction_Status", [254, 255]),
]


def main():
    all_dfs = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for table_id, name, pages in TABLE_DEFS:
            if table_id == 102:
                df = extract_table(pdf, pages, table_id, name, name_list=SUBDIVISION_NAMES, entity_col="Subdivision")
                entity_col = "Subdivision"
            elif table_id == 107:
                df = extract_table_107(pdf, pages, table_id)
                entity_col = "State"
            else:
                df = extract_table(pdf, pages, table_id, name)
                entity_col = "State"
            n_entities = df[entity_col].nunique() if not df.empty else 0
            n_years = df["Fiscal_Year"].nunique() if not df.empty else 0
            print(f"Table {table_id:>3} {name:<32} -> {len(df):>5} rows | {n_entities} {entity_col.lower()}s x {n_years} years")
            if not df.empty:
                out_path = os.path.join(OUT_DIR, f"table_{table_id}_{name}.csv")
                df.to_csv(out_path, index=False)
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "_all_tables_long.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined: {len(combined)} rows -> {combined_path}")


if __name__ == "__main__":
    main()
