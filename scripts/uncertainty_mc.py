"""
Monte Carlo propagation of emission-factor uncertainty through the bottom-up target (v3).

Each draw multiplies the components by NATIONAL-LEVEL (fully correlated across states) factors, representing systematic
error in the factors. Interval widths below are ASSUMED, in the spirit of IPCC 2006 Vol.1 Ch.3 Approach 2 and the
factor ranges in Vol.4; THEY MUST BE VERIFIED AGAINST THE IPCC TABLES BEFORE RELYING ON THE RESULT. Treat the output as a
sensitivity analysis, not a formal inventory uncertainty statement. Activity-data uncertainty is included only through the distance-based terms above.

ACTIVITY-DATA uncertainty (added v3.3, also ASSUMED): where livestock or fertiliser values are interpolated or held constant,
each state gets a draw-specific multiplier 1 + sigma*z (z ~ N(0,1), shared across that state's years, independent across
states) with sigma growing with the distance in years from the nearest real observation:
  livestock   sigma = min(0.15, 0.010 x years from nearest census year)   observed census years: 0
  fertiliser  state pattern not observed: min(0.25, 0.03 x years from the nearest observed (2013-2021) year); observed: 0
Component multipliers (distribution; 90% interval):
  rice CH4            triangular(0.65, 1.00, 1.50)     IPCC rice EF range is roughly -40%/+70% of the default
  livestock CH4       triangular(0.80, 1.00, 1.30)     Tier-2 enteric ~ +/-20-30%
  livestock N2O       triangular(0.50, 1.00, 2.00)     small component
  fertiliser direct   lognormal, 5-95% = [0.5, 2.0]    IPCC EF1 default range spans x0.3 - x3
  fertiliser indirect triangular(0.50, 1.00, 1.60)     volatilisation / leaching fractions
  N share of NPK      uniform(0.93, 1.07)              residual after calibrating the share on 2014-2018 overlap data
  residue burning     triangular(0.60, 1.00, 1.60)
Outputs (results/state_panel/): uncertainty_mc_national.csv, uncertainty_mc_state_year.csv, uncertainty_mc_ghgpi_coverage.csv
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results/state_panel")
N = 2000
rng = np.random.default_rng(42)

p = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv"))
fac = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/india_specific_emission_factors.csv"))
fac = dict(zip(fac.parameter, fac.value))
direct = fac["fert_direct_ef"]
indirect = fac["fert_frac_volatilised"] * fac["fert_ef_deposition"] + fac["fert_frac_leached"] * fac["fert_ef_leaching"]
share_direct = direct / (direct + indirect)

rice = p.GHG_Rice_CH4_ktCO2e.fillna(0).values
lv_ch4 = p.GHG_Livestock_CH4_ktCO2e.fillna(0).values
lv_n2o = p.GHG_Livestock_N2O_ktCO2e.fillna(0).values
fert = p.GHG_FertilizerSoil_N2O_ktCO2e.fillna(0).values
res = (p.GHG_ResidueBurning_CH4_ktCO2e.fillna(0) + p.GHG_ResidueBurning_N2O_ktCO2e.fillna(0)).values
valid = p.Target_Agri_GHG_ktCO2e.notna().values


# ---- activity-data uncertainty by distance from the nearest real observation ----
st_codes, st_names = pd.factorize(p.State)
yrs = p.Year_Standardized.values
census = {}
for ayear, fname in ((2007, "livestock_census_2007.csv"), (2012, "livestock_census_2012.csv"), (2019, "livestock_census_2019.csv")):
    pth = os.path.join(BASE, "data/processed/state_panel", fname)
    if os.path.exists(pth):
        for nm in pd.read_csv(pth)["State"]:
            census.setdefault(nm, []).append(ayear)
dist_l = np.array([min([abs(y - a) for a in census.get(nm, [])] or [np.nan]) for nm, y in zip(p.State, yrs)], dtype=float)
sig_l = np.where(np.isnan(dist_l), 0.0, np.minimum(0.15, 0.010 * dist_l))
fl = p.Flag_Fertiliser.values
obs_years = {nm: p.loc[(p.State == nm) & (p.Flag_Fertiliser == "observed"), "Year_Standardized"].values for nm in p.State.unique()}
dist_f = np.array([min([abs(y - o) for o in obs_years[nm]] or [np.nan]) for nm, y in zip(p.State, yrs)], dtype=float)
# national level is official; the uncertainty is the STATE PATTERN, which ages with distance from the nearest observed year
sig_f = np.where(fl == "observed", 0.0, np.where(np.isnan(dist_f), 0.0, np.minimum(0.25, 0.03 * np.maximum(1, dist_f))))
n_states = len(st_names)
z_l, z_f = rng.standard_normal((N, n_states)), rng.standard_normal((N, n_states))
act_l = np.clip(1 + sig_l[None, :] * z_l[:, st_codes], 0.3, None)   # (N, rows)
act_f = np.clip(1 + sig_f[None, :] * z_f[:, st_codes], 0.3, None)


def tri(lo, mode, hi):
    return rng.triangular(lo, mode, hi, N)


m_rice, m_lvc, m_lvn = tri(0.65, 1, 1.5), tri(0.8, 1, 1.3), tri(0.5, 1, 2.0)
sigma = np.log(2.0 / 0.5) / (2 * 1.645)
m_fd = np.exp(rng.normal(0.0, sigma, N)) * 1.0          # median 1, 5-95% = [0.5, 2.0]
m_fi, m_res = tri(0.5, 1, 1.6), tri(0.6, 1, 1.6)
m_n = rng.uniform(0.93, 1.07, N)   # N share is now calibrated on observed data; +/-7% residual

draws = (rice[None, :] * m_rice[:, None] + lv_ch4[None, :] * m_lvc[:, None] * act_l + lv_n2o[None, :] * m_lvn[:, None] * act_l
         + fert[None, :] * (share_direct * m_fd[:, None] + (1 - share_direct) * m_fi[:, None]) * m_n[:, None] * act_f
         + res[None, :] * m_res[:, None])                    # (N, rows)
draws[:, ~valid] = np.nan

q = lambda a, pct: np.nanpercentile(a, pct, axis=0)
state = p[["State", "Year_Standardized"]].copy()
state["target_central"] = p.Target_Agri_GHG_ktCO2e
state["p05"], state["p50"], state["p95"] = q(draws, 5), q(draws, 50), q(draws, 95)
state.to_csv(os.path.join(RES, "uncertainty_mc_state_year.csv"), index=False)

years = sorted(p.Year_Standardized.unique())
nat = []
for y in years:
    idx = (p.Year_Standardized == y).values & valid
    tot = np.nansum(draws[:, idx], axis=1) / 1000.0
    nat.append({"Year": y, "central_Mt": p.Target_Agri_GHG_ktCO2e[idx].sum() / 1000, "p05_Mt": np.percentile(tot, 5),
                "p50_Mt": np.percentile(tot, 50), "p95_Mt": np.percentile(tot, 95)})
nat = pd.DataFrame(nat)
nat.to_csv(os.path.join(RES, "uncertainty_mc_national.csv"), index=False)
print(nat[nat.Year.isin([2005, 2010, 2015, 2018, 2020])].round(1).to_string(index=False))
r20 = nat[nat.Year == 2020].iloc[0]
print(f"2020: central {r20.central_Mt:.0f} Mt, 90% interval {r20.p05_Mt:.0f}-{r20.p95_Mt:.0f} Mt; BUR4 = 406 Mt "
      f"({'inside' if r20.p05_Mt <= 406 <= r20.p95_Mt else 'outside'})")

g = pd.read_csv(os.path.join(BASE, "data/processed/state_panel/ghgpi_state_agri_2005_2018.csv"))
g["ref"] = g.rice_ktCO2e + g.enteric_ktCO2e + g.manure_ktCO2e + g.soils_ktCO2e + g.cropland_burning_ktCO2e
c = state.merge(g[["State", "Year_Standardized", "ref"]], on=["State", "Year_Standardized"]).dropna(subset=["p05", "ref"])
c = c[c.ref > 0]
# Andhra Pradesh / Telangana: GHGPI keeps them undivided before 2014 while the RBI series splits them from 2010, so those state-years
# are not comparable one by one and are left out of the state-year coverage figure (the national comparison is unaffected).
c = c[~(c.State.isin(["Andhra Pradesh", "Telangana"]) & (c.Year_Standardized < 2014))]
inside = (c.ref >= c.p05) & (c.ref <= c.p95)
cov = pd.DataFrame([{"n_state_years": len(c), "share_GHGPI_inside_90pct_interval": inside.mean(),
                     "median_relative_interval_width": ((c.p95 - c.p05) / c.p50).median()}])
cov.to_csv(os.path.join(RES, "uncertainty_mc_ghgpi_coverage.csv"), index=False)
print(cov.round(3).to_string(index=False))
nat_gh = g.groupby("Year_Standardized").ref.sum() / 1000
nn = nat.set_index("Year").join(nat_gh.rename("GHGPI_Mt"), how="inner")
print("national years with GHGPI inside the 90% interval:", int(((nn.GHGPI_Mt >= nn.p05_Mt) & (nn.GHGPI_Mt <= nn.p95_Mt)).sum()), "of", len(nn))
