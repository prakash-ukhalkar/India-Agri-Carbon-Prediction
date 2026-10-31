"""
Automated checks for the properties the documented claims rest on.
Run from the project root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import validation_suite as vs  # noqa: E402
from _panel_io import upsert_into_panel  # noqa: E402

PANEL = os.path.join(ROOT, "data/processed/state_panel/state_year_panel_with_target.csv")
TARGET = "Target_Agri_GHG_ktCO2e"


def synthetic_panel(seed=0, n_states=8, n_years=14):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_states):
        size = rng.uniform(50, 500)
        for t in range(n_years):
            f = rng.normal(size=5)
            rows.append({"State": f"S{s}", "Year_Standardized": 2004 + t, "Gross_Sown_Area": size * (1 + 0.01 * t),
                         "T": size * 3 * (1 + 0.01 * t) + f[0] * 5, **{f"F__x{i}": f[i] for i in range(5)}})
    df = pd.DataFrame(rows)
    df["T__lag1"] = df.groupby("State")["T"].shift(1)
    return df


class PanelIO(unittest.TestCase):
    def test_upsert_idempotent_and_order_independent(self):
        panel = pd.DataFrame({"State": ["A", "A", "B"], "Year_Standardized": [2012, 2019, 2012], "v": [1.0, np.nan, 3.0]})
        n1 = pd.DataFrame({"State": ["A"], "Year_Standardized": [2019], "v": [9.0]})
        n2 = pd.DataFrame({"State": ["B"], "Year_Standardized": [2012], "v": [7.0]})
        a = upsert_into_panel(upsert_into_panel(panel, n1), n2)
        b = upsert_into_panel(upsert_into_panel(panel, n2), n1)
        c = upsert_into_panel(a, n1)
        for x in (a, b, c):
            self.assertFalse([col for col in x.columns if col.endswith(("_x", "_y", "__new"))])
        pd.testing.assert_frame_equal(a.sort_values(["State", "Year_Standardized"]).reset_index(drop=True),
                                      b.sort_values(["State", "Year_Standardized"]).reset_index(drop=True))
        pd.testing.assert_frame_equal(a, c)
        self.assertEqual(a.loc[(a.State == "A") & (a.Year_Standardized == 2019), "v"].iloc[0], 9.0)
        self.assertEqual(a.loc[(a.State == "A") & (a.Year_Standardized == 2012), "v"].iloc[0], 1.0)


class Leakage(unittest.TestCase):
    def test_selection_uses_only_training_rows(self):
        df = synthetic_panel()
        cols = [c for c in df.columns if c.startswith("F__")]
        train = df[df.State != "S0"]
        before = vs.select(train, cols, "T")
        df2 = df.copy()
        df2.loc[df2.State == "S0", cols + ["T"]] = 1e6      # corrupt the held-out state
        after = vs.select(df2[df2.State != "S0"], cols, "T")
        self.assertEqual(before, after)

    def test_loso_prediction_ignores_heldout_target(self):
        df = synthetic_panel()
        cols = [c for c in df.columns if c.startswith("F__")]
        _, P1, _ = vs.run(df, cols, "T", "LOSO")
        df2 = df.copy()
        df2.loc[df2.State == "S3", "T"] = df2.loc[df2.State == "S3", "T"] * 100  # change only S3's labels
        _, P2, _ = vs.run(df2, cols, "T", "LOSO")
        meta = vs.run.meta
        mask = (meta.State == "S3").values
        for name in ("RF", "XGB", "base_scale_only"):
            np.testing.assert_allclose(P1[name][mask], P2[name][mask], rtol=1e-9,
                                       err_msg=f"{name} prediction for a held-out state depends on its own target")

    def test_expanding_window_never_trains_on_future(self):
        df = synthetic_panel(n_years=16)
        cols = [c for c in df.columns if c.startswith("F__")]
        _, P1, _ = vs.run(df, cols, "T", "Expanding")
        meta = vs.run.meta.copy()
        df2 = df.copy()
        last = df2.Year_Standardized.max()
        df2.loc[df2.Year_Standardized == last, "T"] *= 100   # change only the final year's labels
        _, P2, _ = vs.run(df2, cols, "T", "Expanding")
        early = (meta.Year_Standardized < last).values
        for name in ("RF", "XGB", "base_own_history_mean"):
            np.testing.assert_allclose(P1[name][early], P2[name][early], rtol=1e-9)


class RbiParser(unittest.TestCase):
    """Regression tests for defects found by verify_rbi_extraction.py (2026-10-31)."""

    @classmethod
    def setUpClass(cls):
        import extract_rbi_handbook as ex
        cls.ex = ex

    def rows(self, text):
        return {n: v for n, v in self.ex.parse_page(text)}

    def test_footnote_asterisk_value_is_kept(self):
        r = self.rows("Gujarat 9,747 9,722 10114*" + chr(10) + "Haryana 3,527 3,566 3,556")
        self.assertEqual(r["Gujarat"], [9747.0, 9722.0, 10114.0])

    def test_hash_marker_after_name_does_not_shift_values(self):
        r = self.rows("Andhra Pradesh# 8.53 0.89 8.57")
        self.assertEqual(r["Andhra Pradesh"], [8.53, 0.89, 8.57])

    def test_wrapped_name_is_joined(self):
        r = self.rows("Andaman & Nicobar" + chr(10) + "0.33 0.01 0.29" + chr(10) + "Islands" + chr(10) + "Andhra Pradesh 8.5 0.9 8.6")
        self.assertEqual(r["Andaman & Nicobar Islands"], [0.33, 0.01, 0.29])
        self.assertEqual(r["Andhra Pradesh"], [8.5, 0.9, 8.6])

    def test_indian_digit_grouping_and_dash(self):
        r = self.rows("Tamil Nadu 1,18,518 - 2,02,000")
        self.assertEqual(r["Tamil Nadu"], [118518.0, None, 202000.0])

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "results/state_panel/rbi_extraction_check.csv")), "run scripts/verify_rbi_extraction.py")
    def test_extracted_tables_sum_to_the_pdfs_all_india_rows(self):
        r = pd.read_csv(os.path.join(ROOT, "results/state_panel/rbi_extraction_check.csv"))
        key = r[r.variable.isin(["Rice_Area", "Rice_Production", "Wheat_Production", "Coarse_Cereals_Production", "Fertiliser_Consumption_NPK"])]
        self.assertLess(key.dev.abs().max(), 0.03, "emission-input tables should sum to the PDF totals within 3%")
        self.assertGreater((r.dev.abs() < 0.05).mean(), 0.97)


class Baselines(unittest.TestCase):
    def test_scale_only_is_monotone_in_size(self):
        df = synthetic_panel()
        train, test = df[df.State != "S0"], df[df.State == "S0"].copy()
        small, large = test.copy(), test.copy()
        small["Gross_Sown_Area"], large["Gross_Sown_Area"] = 10.0, 1000.0
        self.assertLess(vs.scale_pred(train, small, "T").mean(), vs.scale_pred(train, large, "T").mean())


@unittest.skipUnless(os.path.exists(PANEL), "panel with target not built")
class RealPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = pd.read_csv(PANEL)

    def test_unique_keys_and_coverage(self):
        self.assertEqual(self.p.duplicated(["State", "Year_Standardized"]).sum(), 0)
        self.assertEqual(self.p.State.nunique(), 37)

    def test_target_is_sum_of_components_and_excludes_electricity(self):
        comps = [c for c in self.p.columns if c.startswith("GHG_")]
        self.assertFalse([c for c in comps if "Electricity" in c], "electricity must not be in the target")
        np.testing.assert_allclose(self.p[comps].sum(axis=1, min_count=1).fillna(0), self.p[TARGET].fillna(0), rtol=1e-9)

    def test_national_total_close_to_bur4(self):
        total = self.p[self.p.Year_Standardized == 2020][TARGET].sum() / 1000
        self.assertTrue(0.8 * 406 <= total <= 1.2 * 406, f"2020 total {total:.0f} Mt outside 0.8-1.2x of BUR4 (406 Mt)")

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data/processed/state_panel/ghgpi_state_agri_2005_2018.csv")),
                         "GHGPI benchmark not built")
    def test_agrees_with_ghgpi_benchmark(self):
        g = pd.read_csv(os.path.join(ROOT, "data/processed/state_panel/ghgpi_state_agri_2005_2018.csv"))
        m = self.p.merge(g, on=["State", "Year_Standardized"])
        ref = m.rice_ktCO2e + m.enteric_ktCO2e + m.manure_ktCO2e + m.soils_ktCO2e + m.cropland_burning_ktCO2e
        ok = m[TARGET].notna() & (ref > 0)
        ratio = m.loc[ok, TARGET].sum() / ref[ok].sum()
        self.assertTrue(0.9 <= ratio <= 1.1, f"national ratio to GHGPI {ratio:.2f} outside 0.9-1.1")
        mape = ((m.loc[ok, TARGET] - ref[ok]).abs() / ref[ok]).median()
        self.assertLess(mape, 0.15, f"median state-year error vs GHGPI {mape:.1%}")

    def test_features_exclude_formula_inputs_and_target_lags(self):
        _, cols = vs.build()
        base = {c[3:] for c in cols}
        for bad in vs.EXCLUDE:
            self.assertFalse(any(b == bad or b.startswith(bad + "_") for b in base), f"{bad} leaked into features")
        self.assertFalse([c for c in cols if c.startswith(("F__Target", "F__GHG_", "F__Livestock_", "F__Memo_"))])

    def test_no_candidate_feature_is_a_near_exact_function_of_a_target_component(self):
        """Name-independent leakage guard: any candidate feature that is (almost) perfectly correlated with a target
        component or the target itself is probably a formula input, whatever it is called."""
        df, cols = vs.build()
        targets = [c for c in df.columns if c.startswith("GHG_")] + [vs.LEVEL, "Target_Dynamic"]
        offenders = []
        for f in cols:
            for t_ in targets:
                x = df[[f, t_]].dropna()
                if len(x) > 50 and x[f].std() > 0 and x[t_].std() > 0 and abs(np.corrcoef(x[f], x[t_])[0, 1]) > 0.995:
                    offenders.append((f, t_))
        self.assertFalse(offenders, f"possible formula inputs among features: {offenders[:5]}")


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data_release", "agri_ghg_state_year.csv")), "release not built")
class Release(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = pd.read_csv(os.path.join(ROOT, "data_release", "agri_ghg_state_year.csv"))
        cls.dic = pd.read_csv(os.path.join(ROOT, "data_release", "data_dictionary.csv"))

    def test_every_column_documented_and_keys_unique(self):
        self.assertEqual(set(self.d.columns), set(self.dic["column"]))
        self.assertEqual(self.d.duplicated(["State", "Year_Standardized"]).sum(), 0)

    def test_flags_use_documented_values(self):
        allowed = {"Flag_Livestock": {"observed", "interpolated", "held_constant", "missing"},
                   "Flag_Fertiliser": {"observed", "national_downscaled", "held_constant", "missing"},
                   "Flag_Rice": {"observed", "missing"},
                   "Flag_Residue": {"observed", "structural_zero_assumed", "missing"},
                   "Data_Completeness": {"complete", "partial"}}
        for col, ok in allowed.items():
            self.assertTrue(set(self.d[col].dropna()) <= ok, col)

    def test_interval_brackets_central_estimate_and_complete_rows_have_all_components(self):
        v = self.d.dropna(subset=["Total_p05_ktCO2e", "Target_Agri_GHG_ktCO2e"])
        self.assertGreater((v.Total_p95_ktCO2e >= v.Total_p05_ktCO2e).mean(), 0.999)
        comp = [c for c in self.d.columns if c.startswith("GHG_")]
        full = self.d[self.d.Data_Completeness == "complete"]
        self.assertFalse(full[comp].isna().any().any())
        self.assertTrue(((self.d.Target_Observed_Share.dropna() >= 0) & (self.d.Target_Observed_Share.dropna() <= 1.0001)).all())


class SustainabilitySummary(unittest.TestCase):
    """Intensity and decoupling outputs are internally consistent (run scripts/sustainability_analysis.py first)."""

    def setUp(self):
        f = os.path.join(ROOT, "results", "state_panel", "sustainability_state_summary.csv")
        if not os.path.exists(f):
            self.skipTest("run scripts/sustainability_analysis.py")
        self.s = pd.read_csv(f)

    def test_shares_sum_to_100_and_sources_to_1(self):
        self.assertAlmostEqual(self.s.share_of_national_pct.sum(), 100, places=3)
        src = self.s[[c for c in self.s if c.startswith("share_") and c != "share_of_national_pct"]].sum(axis=1)
        self.assertTrue(((src - 1).abs() < 1e-3).all())

    def test_elasticity_matches_growth_rates_and_flat_gsva_is_unclassified(self):
        ok = self.s.elasticity.notna()
        self.assertTrue(np.allclose(self.s.elasticity[ok], self.s.g_E_pct_yr[ok] / self.s.g_V_pct_yr[ok], atol=1e-2))
        flat = self.s.g_V_pct_yr.abs() < 0.5
        self.assertTrue((self.s["class"][flat] == "no clear GSVA trend").all())

    def test_national_index_starts_at_100(self):
        i = pd.read_csv(os.path.join(ROOT, "results", "state_panel", "sustainability_national_indices.csv"), index_col=0)
        self.assertTrue(np.allclose(i.iloc[0], 100))


class SustainabilityExtensions(unittest.TestCase):
    """Component and window outputs are consistent with the main summary (run scripts/sustainability_extensions.py first)."""

    def setUp(self):
        d = os.path.join(ROOT, "results", "state_panel")
        if not os.path.exists(os.path.join(d, "sustainability_components_national.csv")):
            self.skipTest("run scripts/sustainability_extensions.py")
        self.n = pd.read_csv(os.path.join(d, "sustainability_components_national.csv"))
        self.w = pd.read_csv(os.path.join(d, "sustainability_window_units.csv"))
        self.s = pd.read_csv(os.path.join(d, "sustainability_state_summary.csv"))

    def test_component_shares_sum_to_about_100(self):
        parts = self.n[self.n.component != "All components"].share_2021_23_pct.sum()
        self.assertAlmostEqual(parts, 100, delta=1.0)

    def test_full_window_class_matches_main_summary(self):
        m = self.w.merge(self.s[["State", "class"]], on="State")
        self.assertTrue((m["class_2011_2023"] == m["class"]).all())

    def test_national_elasticity_is_growth_ratio(self):
        a = self.n[self.n.component == "All components"].iloc[0]
        self.assertAlmostEqual(a.elasticity, a.g_pct_yr / a.g_GSVA_pct_yr, places=2)


if __name__ == "__main__":
    unittest.main()
