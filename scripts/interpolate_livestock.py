"""
Build annual livestock headcounts for every panel row from the census anchor files (rebuilt from scratch each run, so the
step is idempotent and never treats earlier interpolated values as observations).

Anchors (per state and species, wherever present):
  2007  livestock_census_2007.csv   (secondary compilation, see parse_livestock_census_2007.py)
  2012  livestock_census_2012.csv   (official report, our parsing)
  2019  livestock_census_2019.csv   (official report, our parsing)
Policy:
  - between two anchors: linear interpolation (a real estimate bounded by two measurements);
  - before the first / after the last anchor: held constant at the nearest anchor (no trend extrapolation);
  - never borrowed across states or species; a species with fewer anchors (e.g. mules, yaks, mithun have only 2012/2019)
    simply uses the anchors it has.
Writes the Livestock_* columns of state_year_panel.csv for all rows.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(BASE, "data", "processed", "state_panel")
PANEL_PATH = os.path.join(SP, "state_year_panel.csv")
ANCHORS = {2007: "livestock_census_2007.csv", 2012: "livestock_census_2012.csv", 2019: "livestock_census_2019.csv"}


def load_anchors():
    out = {}
    for year, fname in ANCHORS.items():
        path = os.path.join(SP, fname)
        if os.path.exists(path):
            out[year] = pd.read_csv(path).set_index("State")
    return out


AP, TG = "Andhra Pradesh", "Telangana"


def ap_telangana_series(panel, anchors, col):
    """Annual values for Andhra Pradesh and Telangana for one species, boundary-consistent.
    The 2007 and 2012 censuses report Andhra Pradesh UNDIVIDED (Telangana included); 2019 reports them separately. The
    RBI crop tables split the two states from the first year Telangana has a row (2010). So: interpolate the COMBINED
    series between censuses, keep Andhra Pradesh undivided before that year, and from then on split the combined series
    with Telangana's observed 2019 share (held constant). The sum is therefore always the interpolated combined total."""
    pts = {}
    for y, a in anchors.items():
        if col not in a.columns or AP not in a.index or pd.isna(a.loc[AP, col]):
            continue
        if y >= 2019:
            if TG in a.index and pd.notna(a.loc[TG, col]):
                pts[y] = a.loc[AP, col] + a.loc[TG, col]      # divided census: combine
        else:
            pts[y] = a.loc[AP, col]                            # undivided census
    if not pts:
        return None, None
    ay = np.array(sorted(pts)); av = np.array([pts[y] for y in ay], dtype=float)
    a19 = anchors.get(2019)
    share = 0.0
    if a19 is not None and TG in a19.index and AP in a19.index and col in a19.columns:
        tot = a19.loc[AP, col] + a19.loc[TG, col]
        share = a19.loc[TG, col] / tot if tot else 0.0
    tel_first = panel.loc[panel.State == TG, "Year_Standardized"].min()
    out = {}
    for st in (AP, TG):
        years = panel.loc[panel.State == st, "Year_Standardized"].values
        comb = np.interp(years, ay, av)
        if st == AP:
            vals = np.where(years >= tel_first, comb * (1 - share), comb)
        else:
            vals = comb * share
        out[st] = pd.Series(vals, index=panel.index[panel.State == st])
    return out[AP], out[TG]


def main():
    panel = pd.read_csv(PANEL_PATH).sort_values(["State", "Year_Standardized"]).reset_index(drop=True)
    anchors = load_anchors()
    cols = sorted({c for a in anchors.values() for c in a.columns if c.startswith("Livestock_")})
    for col in cols:
        panel[col] = np.nan
    for col in cols:
        ap_s, tg_s = ap_telangana_series(panel, anchors, col)
        if ap_s is not None:
            panel.loc[ap_s.index, col] = ap_s.values
            panel.loc[tg_s.index, col] = tg_s.values
    for state, idx in panel.groupby("State").groups.items():
        if state in (AP, TG):
            continue
        years = panel.loc[idx, "Year_Standardized"].values
        for col in cols:
            pts = {y: a.loc[state, col] for y, a in anchors.items()
                   if state in a.index and col in a.columns and pd.notna(a.loc[state, col])}
            if not pts:
                continue
            ay = np.array(sorted(pts))
            av = np.array([pts[y] for y in ay], dtype=float)
            panel.loc[idx, col] = np.interp(years, ay, av)  # linear inside; constant outside (np.interp clamps)
    panel.to_csv(PANEL_PATH, index=False)
    cov = panel[cols[0]].notna().mean()
    print(f"Livestock columns rebuilt from anchors {sorted(anchors)}; coverage {cov:.1%} of {len(panel)} rows")
    print("\nNational cattle (millions), selected years:")
    nat = panel.groupby("Year_Standardized")["Livestock_Cattle_Head"].sum() / 1e6
    print(nat.loc[[2004, 2007, 2010, 2012, 2015, 2019, 2024]].round(1).to_string())


if __name__ == "__main__":
    main()
