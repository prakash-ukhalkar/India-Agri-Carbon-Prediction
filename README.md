# State-Level Agricultural GHG Emissions in India

Author: Prakash Ukhalkar. Code: MIT. Compiled dataset values: CC BY 4.0.

A reproducible state-by-year inventory (37 states/UTs, 2004-2024, 748 state-years) of Indian agricultural
CH4 and N2O emissions in kt CO2e, built bottom-up with IPCC-style methods from official statistics:
rice cultivation, livestock (enteric fermentation and manure), fertiliser soil N2O, and crop-residue burning.
Every value carries per-component data-quality flags, and a Monte Carlo uncertainty range is provided.

Archived releases:
- Dataset: https://doi.org/10.5281/zenodo.23836003
- Code: https://doi.org/10.5281/zenodo.23836551

## Validation in brief
- National totals are 0.84-0.86 of India's BUR4 agriculture series in 2011-2019, with the same trend.
- Against the GHG Platform India state inventory (490 unit-years), the total ratio is 1.00 and the median
  state-year error is 6.3%. Andhra Pradesh and Telangana are compared as one unit.
- Source extraction is checked against the All-India totals printed in the source tables.
- A leakage-free machine-learning check is included as a secondary analysis (`results/state_panel/`).

## Layout
```text
data/raw/README_sources.md                 where each source comes from (the raw files themselves are not redistributed)
data/processed/state_panel/                parsed source tables and the panel (state_year_panel_with_target.csv)
scripts/                                   parsing, target construction, validation, uncertainty
results/state_panel/                       validation outputs
notebooks/01_state_panel_walkthrough.ipynb reads the panel and saved results and reproduces the tables and figures
data_release/                              the public dataset, dictionary, emission factors and checksums
tests/                                     automated checks (leakage, baselines, target definition, idempotent merges)
```

## Pipeline (Python 3.10)
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/interpolate_livestock.py            # annual livestock from the census anchors
python scripts/build_tier1_ghg_target.py           # emissions (GWP_SET=AR5 for AR5 GWPs; default SAR, as in the BUR)
python scripts/reconcile_with_bur.py               # national check against the official inventory
python scripts/compare_with_ghgpi.py               # state-year agreement with the benchmark
python scripts/uncertainty_mc.py                   # Monte Carlo emission-factor uncertainty
python scripts/validation_suite.py                 # nested spatial + temporal CV, all targets, all baselines
python scripts/sustainability_analysis.py          # intensity, decoupling and source profiles by state (2011-2023)
python scripts/sustainability_extensions.py        # decoupling by component and sensitivity to the time window
python scripts/make_figures.py                     # figures (results/figures)
python scripts/build_data_release.py               # data_release/
```
Parsing steps need the raw source files listed in `data/raw/README_sources.md`. The panel-merge scripts are
idempotent (`scripts/_panel_io.py`).

## Known limitations
No independent state-level ground truth. Livestock is interpolated between census years and fertiliser is
downscaled from national totals before 2013; both are flagged per value. Emission-factor values are listed
with their verification status in `data_release/emission_factors.csv`.
