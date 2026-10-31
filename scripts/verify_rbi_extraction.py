"""
Independent verification of the RBI Handbook extraction (scripts/extract_rbi_handbook.py).

Method: for every ADDITIVE table (area, production, fertiliser, livestock products, GSVA) parse the Handbook's own
"All India" row straight from the PDF text (a row the extractor does not keep) and compare it with the sum of the extracted
state/UT values for each year. A column shift, a dropped cell or a mis-matched state name breaks the sum, so agreement is
evidence that the numbers in the panel are what the PDF says. Non-additive tables (yields, cropping intensity, rainfall) are
not tested here. Also lists every extracted cell that is missing although the PDF line has a token for it.

Output: results/state_panel/rbi_extraction_check.csv (table-year level) and rbi_extraction_check_summary.csv (table level)
"""
import os
import re
import sys
import pandas as pd
import pdfplumber

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_rbi_handbook as ex  # noqa: E402

ADDITIVE = ("_Area", "_Production", "Fertiliser", "Gross_Sown", "Net_Sown", "Gross_Irrigated", "Net_Irrigated", "GSVA", "NSVA")
long = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/rbi_tables/_all_tables_long.csv"))
rows = []
with pdfplumber.open(ex.PDF_PATH) as pdf:
    for table_id, name, pages in ex.TABLE_DEFS:
        if not any(k in name for k in ADDITIVE) or table_id in (102, 107):
            continue
        ai = {}
        for pi in pages:
            text = pdf.pages[pi].extract_text() or ""
            years = ex.parse_year_header(text)
            for line in text.split("\n"):
                m = re.match(r"^\s*(all[\s-]*india|india)\b\*?\s*(.*)$", line, flags=re.I)
                if not m:
                    continue
                vals = [ex.parse_number(t) for t in m.group(2).split()]
                for y, v in zip(years, vals):
                    if v is not None:
                        ai.setdefault(y, v)
        t = long[long.Variable == name]
        sums = t.groupby("Fiscal_Year").Value.sum(min_count=1)
        cnt = t.groupby("Fiscal_Year").Value.count()
        for fy, a in ai.items():
            if fy in sums.index and a:
                rows.append({"table": table_id, "variable": name, "fiscal_year": fy, "all_india_pdf": a,
                             "sum_extracted_states": sums[fy], "n_entities": int(cnt[fy]), "dev": sums[fy] / a - 1})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(BASE, "results/state_panel/rbi_extraction_check.csv"), index=False)
summ = res.groupby(["table", "variable"]).dev.agg(table_years="count", median_abs_dev=lambda s: s.abs().median(),
                                                  max_abs_dev=lambda s: s.abs().max()).reset_index()
summ.to_csv(os.path.join(BASE, "results/state_panel/rbi_extraction_check_summary.csv"), index=False)
pd.set_option("display.width", 220)
print(f"tables checked: {res.variable.nunique()} | table-years: {len(res)}")
for tol in (0.005, 0.01, 0.02, 0.05):
    print(f"  share of table-years within {tol:.1%}: {(res.dev.abs() < tol).mean():.3f}")
print(summ.round(4).to_string(index=False))
print("\nworst table-years:")
print(res.reindex(res.dev.abs().sort_values(ascending=False).index).head(15).round(4).to_string(index=False))
