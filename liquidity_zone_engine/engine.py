"""Orchestration layer for the liquidity zone engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.bias import compute_daily_bias
from liquidity_zone_engine.bos_choch import detect_bos_choch_timeline
from liquidity_zone_engine.broker_time import validate_broker_timezone
from liquidity_zone_engine.config import DEFAULT_CONFIG, SUPPORTED_TIMEFRAMES
from liquidity_zone_engine.fvg import detect_fvgs, find_fvg_retest_after_sweep
from liquidity_zone_engine.smc_dashboard import build_smc_zones, calibrate_smc_output, format_smc_zones
from liquidity_zone_engine.sessions import analyze_session_liquidity
from liquidity_zone_engine.structure import annotate_liquidity_roles, detect_swings
from liquidity_zone_engine.sweeps import detect_smc_sweeps
from liquidity_zone_engine.utils import normalize_ohlc
from liquidity_zone_engine.zones import (
    build_fractal_zone_map,
    finalize_zones,
    flatten_zone_map,
)


def _enrich_sweeps_with_fvg(
    data: pd.DataFrame,
    sweeps: list[dict[str, Any]],
    fvgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []

    for sweep in sweeps:
        item = dict(sweep)
        setup_type = "buy" if sweep["type"] == "sell_side_sweep" else "sell"
        bar_index = int(sweep.get("bar_index", -1))

        poi = find_fvg_retest_after_sweep(data, bar_index, setup_type, fvgs)
        item["fvg_poi_valid"] = poi is not None
        if poi is not None:
            item["fvg_poi"] = {
                "type": poi["type"],
                "low": poi["low"],
                "high": poi["high"],
                "index": poi["index"],
            }
        enriched.append(item)

    return enriched


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
    Run SMC institutional trading analysis on OHLC data.

    Controlled output: max 20 swings, 12 zones, 10 sweeps, last 5 BOS, 2 scenarios.
    """
    if timeframe is not None and timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported values: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )

    data = normalize_ohlc(df)
    validate_broker_timezone(data)
    atr = compute_atr(data, period=atr_period)

    swings = detect_swings(df, swing_window=swing_window)
    fvgs = detect_fvgs(data, atr_period=atr_period)
    session = analyze_session_liquidity(data)
    daily_bias = compute_daily_bias(data, fvgs=fvgs)

    bos_timeline = detect_bos_choch_timeline(data, swings)

    # Preliminary zones for sweep linking (rebuilt in calibrate)
    prelim_grouped = build_smc_zones(data, swings, bos_timeline, sweeps=[])
    prelim_flat = format_smc_zones(prelim_grouped)

    sweeps = detect_smc_sweeps(df, swings, prelim_flat)
    sweeps = _enrich_sweeps_with_fvg(data, sweeps, fvgs)

    smc = calibrate_smc_output(data, swings, sweeps, bos_timeline)

    flat_zones = smc["zones"]
    swings = annotate_liquidity_roles(swings, smc["sweeps"], flat_zones)

    grouped_raw = build_fractal_zone_map(data, swings, atr)
    flat_raw = flatten_zone_map(grouped_raw)

    result: dict[str, Any] = {
        "market_summary": smc["market_summary"],
        "liquidity_targets": smc["liquidity_targets"],
        "liquidity_pools": smc["liquidity_pools"],
        "zones": smc["zones_grouped"],
        "zones_flat": smc["zones"],
        "sweeps": smc["sweeps"],
        "swings": swings,
        "bos_choch": smc["bos_choch"],
        "trade_scenarios": smc["trade_scenarios"],
        "fvgs": [
            {
                "type": fvg["type"],
                "low": fvg["low"],
                "high": fvg["high"],
                "center": fvg["center"],
                "index": fvg["index"],
                "timestamp": fvg["timestamp"],
                "filled": fvg["filled"],
            }
            for fvg in fvgs
        ],
        "session": session,
        "daily_bias": daily_bias,
        "validation": {
            "zones_before": len(flat_raw),
            **smc["validation"],
        },
    }

    if timeframe is not None:
        result["timeframe"] = timeframe

    return result
