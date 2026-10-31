"""
Shared helpers for the panel-construction scripts.

upsert_into_panel makes the "merge a new source into state_year_panel.csv" steps
idempotent and independent of run order: for columns that already exist, values
from the new source overwrite only where the new source has a value, and
existing values are kept otherwise. Re-running a script no longer creates _x/_y
duplicate columns, and the 2012 and 2019 livestock scripts can run in any order.
"""
import pandas as pd

KEYS = ["State", "Year_Standardized"]


def upsert_into_panel(panel, new, keys=KEYS):
    new_cols = [c for c in new.columns if c not in keys]
    merged = panel.merge(new, on=keys, how="left", suffixes=("", "__new"))
    for c in new_cols:
        incoming = f"{c}__new"
        if incoming in merged.columns:
            merged[c] = merged[incoming].combine_first(merged[c])
            merged = merged.drop(columns=[incoming])
    return merged
