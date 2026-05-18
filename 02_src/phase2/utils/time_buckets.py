"""Election calendar time buckets for temporal stratification."""
from __future__ import annotations

import pandas as pd


def election_time_bucket(date_val) -> str:
    dt = pd.to_datetime(date_val, errors="coerce")
    if pd.isna(dt):
        return "nodate"
    d = dt.normalize()
    if d < pd.Timestamp("2023-11-25"):
        return "pre_registration"
    if d < pd.Timestamp("2023-12-16"):
        return "post_registration"
    if d < pd.Timestamp("2024-01-13"):
        return "campaign"
    return "election_plus"


def window_time_bucket(sent_dates: list) -> str:
    """Assign window bucket from sentence dates (max date in window)."""
    parsed = [pd.to_datetime(d, errors="coerce") for d in sent_dates if d is not None and str(d).strip()]
    valid = [d for d in parsed if not pd.isna(d)]
    if not valid:
        return "nodate"
    return election_time_bucket(max(valid))
