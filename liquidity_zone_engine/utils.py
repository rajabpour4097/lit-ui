"""Shared dataframe helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.config import REQUIRED_OHLC_COLUMNS


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase OHLC columns and a resolved timestamp series."""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")

    normalized = df.copy()
    normalized.columns = [str(col).lower() for col in normalized.columns]

    missing = [col for col in REQUIRED_OHLC_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    for col in REQUIRED_OHLC_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    ohlc_cols = list(REQUIRED_OHLC_COLUMNS)
    if normalized[ohlc_cols].isna().any().any():
        raise ValueError("OHLC columns contain non-numeric values")

    return normalized


def resolve_timestamp(df: pd.DataFrame, index: int) -> Any:
    """Resolve bar timestamp from a column or index."""
    if "timestamp" in df.columns:
        return pd.Timestamp(df.iloc[index]["timestamp"]).to_pydatetime()
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Timestamp(df.index[index]).to_pydatetime()
    return index
