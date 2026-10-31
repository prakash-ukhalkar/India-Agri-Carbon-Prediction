"""
Consolidated, leakage-free validation suite for the state x year GHG panel.
Supersedes model_panel.py / model_panel_diff.py / model_intensity.py (kept for
history), which fitted feature selection on the full data before CV.

Targets : level, dynamic-only (rice+residue+fertiliser), intensity (per ha
          gross sown area), YoY change in the level target ("diff").
Features: EXOGENOUS ONLY - formula inputs and every target-derived column
          (including the target's own lags) are excluded by construction.
          Candidates = base exogenous vars + lag1/lag2/roll3 of each.
Feature selection (corr>0.90 pruning, VIF<10, median imputation, scaling) is
fitted on the TRAINING fold only.
CV      : LOSO (spatial) and expanding-window (temporal, min 10 train years).
Baselines (matched to scheme): other-states mean; scale-only log-log on gross
          sown area; own-history mean; own lag-1; zero-change / own-mean-change.
Output  : results/state_panel/validation_suite.csv
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(BASE, "data/processed/state_panel/state_year_panel_with_target.csv")
OUT = os.path.join(BASE, "results/state_panel/validation_suite.csv")
LEVEL = "Target_Agri_GHG_ktCO2e"
DYN_COMPONENTS = ["GHG_Rice_CH4_ktCO2e", "GHG_ResidueBurning_CH4_ktCO2e",
                  "GHG_ResidueBurning_N2O_ktCO2e", "GHG_FertilizerSoil_N2O_ktCO2e"]
EXCLUDE = ["Rice_Area", "Rice_Production", "Rice_Yield", "Wheat_Area", "Wheat_Production", "Wheat_Yield",
           "Coarse_Cereals_Area", "Coarse_Cereals_Production", "Coarse_Cereals_Yield",
           "Total_Foodgrains_Area", "Total_Foodgrains_Production", "Total_Foodgrains_Yield",
           "Fertiliser_Consumption_NPK", "Fertiliser_N_applied_ktN", "N_compiled_ktN", "N_share_of_NPK", "N_scale_to_national",
           "Agri_Electricity_Consumption_GWh",
           "AgriElectricity_component_is_2019_only", "NASA_Rainfall_mm"]
MAX_VIF, CORR_T, MIN_TRAIN_YEARS = 10.0, 0.90, 10


def build():
    df = pd.read_csv(PANEL).sort_values(["State", "Year_Standardized"]).reset_index(drop=True)
    df["Target_Dynamic"] = df[DYN_COMPONENTS].sum(axis=1, min_count=1)
    df["Target_Intensity"] = np.where(df.Gross_Sown_Area > 0, df[LEVEL] / df.Gross_Sown_Area, np.nan)
    g = df.groupby("State")
    df["Target_Diff"] = g[LEVEL].diff()
    for t in [LEVEL, "Target_Dynamic", "Target_Intensity"]:
        df[f"{t}__lag1"] = g[t].shift(1)  # for the own-lag baseline only, never a feature
    drop = set(EXCLUDE) | {c for c in df.columns if c.startswith(("GHG_", "Memo_", "Livestock_", "Target"))}
    base = [c for c in df.columns
            if c not in drop and c not in ("State", "Year_Standardized") and df[c].dtype.kind in "fi"]
    feats = {}
    for c in base:
        feats[c] = df[c]
        feats[f"{c}_lag1"] = g[c].shift(1)
        feats[f"{c}_lag2"] = g[c].shift(2)
        feats[f"{c}_roll3"] = g[c].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    F = pd.DataFrame(feats)
    F = F.loc[:, F.notna().any() & (F.std() > 0)]
    df = pd.concat([df, F.add_prefix("F__")], axis=1)
    # common sample: needs the scale variable, a prior year (for diff) and history for lags
    df = df[(df.Gross_Sown_Area > 0) & df[LEVEL].notna() & df.Target_Diff.notna() & (df.Year_Standardized >= 2006)]
    return df.reset_index(drop=True), ["F__" + c for c in F.columns]


def select(train, cols, target):
    X = train[cols]
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    cols = list(X.columns)
    tc = X.corrwith(train[target]).abs().fillna(0)
    corr = X.corr().abs().values
    keep = np.ones(len(cols), bool)
    for i in range(len(cols)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(cols)):
            if keep[j] and corr[i, j] > CORR_T:
                if tc.iloc[i] >= tc.iloc[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    cols = [c for c, k in zip(cols, keep) if k]
    while len(cols) > 2:
        C = np.corrcoef(X[cols].values, rowvar=False)
        vif = np.diag(np.linalg.pinv(C))
        if vif.max() <= MAX_VIF:
            break
        cols.pop(int(vif.argmax()))
    return cols


MODELS = {
    "RF": lambda: RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                        random_state=42, n_jobs=-1),
    "XGB": lambda: XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
                                random_state=42, n_jobs=4),
}


def fit_predict(train, test, cols, target):
    sel = select(train, cols, target)
    med = train[sel].median()
    sc = StandardScaler().fit(train[sel].fillna(med))
    Xtr, Xte = sc.transform(train[sel].fillna(med)), sc.transform(test[sel].fillna(med))
    return {n: f().fit(Xtr, train[target].values).predict(Xte) for n, f in MODELS.items()}, len(sel)


def scale_pred(train, test, target):
    ok = train[target] > 0
    lr = LinearRegression().fit(np.log(train.loc[ok, ["Gross_Sown_Area"]]), np.log(train.loc[ok, target]))
    return np.exp(lr.predict(np.log(test[["Gross_Sown_Area"]])))


def run(df, cols, target, scheme):
    """Return (y_true, {name: pred}, mean #features selected), pooled over folds."""
    ys, P, nsel, meta = [], {}, [], []
    if scheme == "LOSO":
        splits = [(df[df.State != s], df[df.State == s]) for s in df.State.unique()]
    else:
        yrs = sorted(df.Year_Standardized.unique())
        splits = [(df[df.Year_Standardized < y], df[df.Year_Standardized == y]) for y in yrs[MIN_TRAIN_YEARS:]]
    for tr, te in splits:
        if len(te) == 0 or len(tr) < 30:
            continue
        ys.append(te[target].values)
        meta.append(te[["State", "Year_Standardized"]])
        preds, k = fit_predict(tr, te, cols, target)
        nsel.append(k)
        for n, p in preds.items():
            P.setdefault(n, []).append(p)
        if scheme == "LOSO":
            P.setdefault("base_other_states_mean", []).append(np.full(len(te), tr[target].mean()))
            if target != "Target_Diff":
                P.setdefault("base_scale_only", []).append(scale_pred(tr, te, target))
        else:
            sm = tr.groupby("State")[target].mean()
            own = te.State.map(sm).fillna(tr[target].mean()).values
            if target == "Target_Diff":
                P.setdefault("base_zero_change", []).append(np.zeros(len(te)))
                P.setdefault("base_own_mean_change", []).append(own)
            else:
                P.setdefault("base_own_history_mean", []).append(own)
                P.setdefault("base_own_lag1", []).append(te[f"{target}__lag1"].fillna(pd.Series(own, index=te.index)).values)
    run.meta = pd.concat(meta, ignore_index=True)
    return np.concatenate(ys), {k: np.concatenate(v) for k, v in P.items()}, float(np.mean(nsel))


def main():
    df, cols = build()
    print(f"sample rows={len(df)} states={df.State.nunique()} "
          f"years={df.Year_Standardized.min()}-{df.Year_Standardized.max()} candidate features={len(cols)}", flush=True)
    rows, pred_rows = [], []
    for target, label in [(LEVEL, "level"), ("Target_Dynamic", "dynamic_only"),
                          ("Target_Intensity", "intensity"), ("Target_Diff", "yoy_change")]:
        d = df.dropna(subset=[target])
        for scheme in ["LOSO", "Expanding"]:
            y, P, k = run(d, cols, target, scheme)
            pr = run.meta.copy()
            pr["target"], pr["scheme"], pr["actual"] = label, scheme, y
            for name, p in P.items():
                pr[name] = p
            pred_rows.append(pr)
            for name, p in P.items():
                r = {"target": label, "scheme": scheme, "model": name, "n": len(y),
                     "mean_features_selected": np.nan if name.startswith("base_") else k,
                     "R2": r2_score(y, p)}
                if target != "Target_Diff":
                    # tree models can emit negative values for tiny states; floor at the
                    # smallest observed target so the log-scale R2 is defined and fair
                    pos = np.clip(p, y[y > 0].min(), None)
                    r["R2_log"] = r2_score(np.log(y[y > 0]), np.log(pos[y > 0]))
                rows.append(r)
            print(f"done {label}/{scheme}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT.replace("validation_suite.csv", "validation_predictions.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
