"""Build smoke-test doc_id fixture."""
from __future__ import annotations

from typing import List

import pandas as pd

from .camp_genre import derive_camp, infer_genre
from .io import read_parquet


def select_smoke_doc_ids(labeled_path, n_docs: int = 5) -> List[str]:
    df = read_parquet(labeled_path)
    return _pick_doc_ids(df, n_docs)


def select_smoke_doc_ids_from_corpus(csv_path, n_docs: int = 5) -> List[str]:
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        usecols=["doc_id", "_source_file", "source", "speakers", "sentence"],
    )
    df = df.groupby("doc_id", as_index=False).first()
    df["genre"] = df.apply(infer_genre, axis=1)
    df["camp"] = df.apply(lambda r: derive_camp(r, r["genre"]), axis=1)
    df = df[df["genre"].isin(["news", "debate"])]
    return _pick_doc_ids(df, n_docs)


def _pick_doc_ids(df: pd.DataFrame, n_docs: int) -> List[str]:
    picks = []
    for genre in ["debate", "news"]:
        sub = df[df["genre"] == genre]
        for camp in ["DPP", "KMT", "TPP"]:
            csub = sub[sub["camp"] == camp]
            if csub.empty:
                continue
            doc = csub["doc_id"].iloc[0]
            if doc not in picks:
                picks.append(doc)
            if len(picks) >= n_docs:
                return picks[:n_docs]
    for doc in df["doc_id"].unique():
        if doc not in picks:
            picks.append(doc)
        if len(picks) >= n_docs:
            break
    return picks[:n_docs]
