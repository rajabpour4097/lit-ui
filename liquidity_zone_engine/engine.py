"""Orchestration layer for the liquidity zone engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG, SUPPORTED_TIMEFRAMES
from liquidity_zone_engine.structure import detect_swings
from liquidity_zone_engine.sweeps import detect_sweeps
from liquidity_zone_engine.utils import normalize_ohlc
from liquidity_zone_engine.zones import (
    _prepare_zone_candidates,
    _remove_redundant_overlaps,
    filter_quality_zones,
    finalize_zones,
    tag_sub_zones,
)


def full_analysis(
    df: pd.DataFrame,
    *,
    swing_window: int = DEFAULT_CONFIG.swing_window,
    atr_period: int = DEFAULT_CONFIG.atr_period,
    zone_k: float = DEFAULT_CONFIG.zone_k,
    cluster_atr_mult: float = DEFAULT_CONFIG.cluster_atr_mult,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """
    Run full liquidity intelligence analysis on a pre-loaded OHLC dataframe.

    This function does not fetch market data. Pass D1/H4/H1/M15/M5 dataframes
    from your existing MT5 data loader.
    """
    if timeframe is not None and timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported values: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )

    swings = detect_swings(df, swing_window=swing_window)
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=atr_period)

    candidates = _prepare_zone_candidates(
        data,
        atr,
        swings,
        zone_k=zone_k,
        cluster_atr_mult=cluster_atr_mult,
    )

    prelim = finalize_zones(data, candidates, sweeps=[], atr=atr)
    quality_zones, _ = filter_quality_zones(prelim, [])
    quality_zones = _remove_redundant_overlaps(quality_zones)

    sweeps = detect_sweeps(df, quality_zones)
    zones = finalize_zones(data, quality_zones, sweeps, atr)
    zones = [
        {key: value for key, value in zone.items() if not key.startswith("_")}
        for zone in zones
    ]
    zones = tag_sub_zones(_remove_redundant_overlaps(zones))

    if timeframe in ("D1", "H4"):
        for zone in zones:
            zone.pop("role", None)

    result: dict[str, Any] = {
        "zones": zones,
        "swings": swings,
        "sweeps": sweeps,
    }

    if timeframe is not None:
        result["timeframe"] = timeframe

    return result
