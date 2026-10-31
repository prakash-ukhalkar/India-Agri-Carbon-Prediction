# Inventory of data/raw (audited 2026-10-31)

Every file and folder, whether it is used, and why. "National" files cannot validate state values.

## Verification status of each PDF-derived input (details: AUDIT_REPORT.md section I)
- RBI Handbook tables: re-parsed All-India rows vs extracted state sums (`scripts/verify_rbi_extraction.py`): emission inputs within 0.06-2.7%; three parser defects found and fixed.
- Livestock censuses 2012/2019: match an independent compilation exactly except J&K cattle 2019 (+15%, unresolved) and Arunachal (+1%).
- CEA General Review 2019-20 agricultural electricity: equals the PDF total exactly.
- Official fertiliser table: equals RBI N+P+K sums (0.1%) and FAOSTAT N (1.5%).
- BUR-4 Table 2.36: transcribed, internally consistent.

## Used in the state panel / dataset
| Path | Content | Use |
|---|---|---|
| `state_panel/rbi_nsdp/Handbook_2024-25.pdf` | RBI Handbook of Statistics on Indian States | Crop area/production, fertiliser N+P+K (2013-2021), land use, groundwater, GSVA. Its state N+P+K sums equal the official All-India table to 0.1% (2013-2021). |
| `state_panel/livestock_census/` (19th 2012, 20th 2019 reports, `datafile.xls`) | Livestock Censuses | 2012 and 2019 anchors (reproduce official national totals). Two files for 2012 are duplicates. |
| `state_panel/imd_climate/nasa_power_state_climate.csv` | NASA POWER temperature at state centroids | Temperature feature |
| `state_panel/state_boundaries/india_state.geojson` | State polygons (35 features; no Telangana/Ladakh) | Not used in the dataset (could support a map) |
| `Farm_Activity_Data/Fertilizer_use/Fertilizer consumption_1951-2024.pdf` | Official All-India N, P2O5, K2O, 1950-51 to 2023-24 | National N control totals (`parse_fertiliser_national_official.py`); Fertiliser Association of India, Fertiliser Statistics |
| `benchmark_ghgpi/GHGPI-AFOLU-Estimates-2005-to-2018.xlsx` | GHG Platform India state inventory (git-ignored; usage terms unconfirmed) | Validation benchmark; emission factors; 2007 livestock anchor; compiled N pattern for 2004-2012 |
| `benchmark_ghgpi/India BUR-4.pdf` | India's Fourth Biennial Update Report | National agriculture series 2011-2020 (Table 2.36) |

## Used only for cross-checks (national)
| Path | Use |
|---|---|
| `Farm_Activity_Data/Fertilizer_use/Fertilizer Use_FAOSTAT.xls` | FAOSTAT India N/P/K; agrees with the official table within 1.5% (2007-2021) |
| `Farm_Activity_Data/Live_stock/Live_Stock_Manuer_FAOSTAT.xls` | N excretion by species 1961-2023; implied stock trend after 2019 |
| `Emission_Data/Emissions_from_Crops_FAOSTAT.xls` | FAOSTAT Tier-1 rice CH4 (1.50x BUR4 in 2020) |
| `scripts/faostat_crosschecks.py` | produces `results/state_panel/faostat_crosschecks.csv` |

## Examined and NOT usable for the dataset
| Path | Finding |
|---|---|
| `Farm_Activity_Data/Crop_Data/horizontal_crop_vertical_year_report*.xls` (DES district reports) | HTML tables. 2012-2022 has no rice columns and omits Odisha, Sikkim, Dadra & Nagar Haveli, Daman & Diu, Ladakh, Lakshadweep; earlier files cover only 8-28 states with rice in few of them. Only 1 state-year could be compared with the RBI rice area, so it cannot validate or fill it. |
| `Farm_Activity_Data/Crop_Data/Crop yields_India_1997-2020_Kaggle.csv` | Third-party compilation with national-style rainfall/fertiliser/pesticide columns repeated per state; provenance unclear |
| `Emission_Data/Emission_Energy_All_FAOSTAT.xls`, `Emission_from_energy_use_in_Agri_FAOSTAT.xls` | National energy emissions; electricity is excluded from the agricultural total |
| `Energy_and_Resource_Consumption_Data/*`, `Environment_and_Soil_Data/Land/*electricity*` | National proxies and CAGR forecasts (legacy national-series work) |
| `Environment_and_Soil_Data/` (land, location, soil texture, soil nutrient dashboards, weather, humidity) | National or non-emission variables; "proxy" and "projected" files; not needed |
| `Farm_Activity_Data/Machine_usage`, `Pestiside_use` | National; no emission formula uses them |
| `state_panel/cea_agri_electricity/*.pdf` (7 CEA reports) | State agricultural electricity; only the FY2019-20 table was parsed, and electricity is not in the target. Later reports could extend an electricity memo series |

## Empty folders (planned sources that were never added)
`RBI_Handbook/`, `state_panel/cgwb_groundwater/`, `state_panel/des_agri_crops/`, `state_panel/des_agri_fertilizer/`.
State-wise fertiliser (DES/DoF) and groundwater (CGWB) tables would be the most useful additions.
