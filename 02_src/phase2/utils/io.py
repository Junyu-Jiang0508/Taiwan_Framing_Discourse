"""Parquet I/O and frozenset serialization."""
from __future__ import annotations

from typing import Any, FrozenSet, Iterable, List, Union

import numpy as np
import pandas as pd


def l2_set_to_tuple(s: FrozenSet[str]) -> tuple:
    return tuple(sorted(s))


def tuple_to_l2_set(t: Any) -> FrozenSet[str]:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return frozenset()
    if isinstance(t, frozenset):
        return t
    if isinstance(t, set):
        return frozenset(str(x) for x in t)
    if isinstance(t, np.ndarray):
        return frozenset(str(x) for x in t.tolist() if x is not None and str(x).strip())
    if isinstance(t, (list, tuple)):
        return frozenset(str(x) for x in t if x)
    if isinstance(t, str):
        if not t.strip():
            return frozenset()
        return frozenset(x.strip() for x in t.split("|") if x.strip())
    return frozenset()


def union_l2_sets(series: Iterable[Any]) -> FrozenSet[str]:
    out: set = set()
    for item in series:
        out |= set(tuple_to_l2_set(item))
    return frozenset(out)


def read_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path, sort_by: List[str] | None = None) -> None:
    out = df.copy()
    if sort_by:
        cols = [c for c in sort_by if c in out.columns]
        if cols:
            out = out.sort_values(cols).reset_index(drop=True)
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def serialize_l2_column(df: pd.DataFrame, col: str = "l2_set") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].map(lambda x: l2_set_to_tuple(tuple_to_l2_set(x)))
    return out


def deserialize_l2_column(df: pd.DataFrame, col: str = "l2_set") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].map(tuple_to_l2_set)
    return out
