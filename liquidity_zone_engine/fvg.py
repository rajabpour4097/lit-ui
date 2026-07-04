"""Fair Value Gap (FVG) detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc, resolve_timestamp


def detect_fvgs(df: pd.DataFrame, atr_period: int | None = None) -> list[dict[str, Any]]:
    """
    Detect Fair Value Gaps (3-candle imbalance).

    Bullish FVG: low of candle 3 > high of candle 1.
    Bearish FVG: high of candle 3 < low of candle 1.
    """
    period = atr_period or DEFAULT_CONFIG.atr_period
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=period)

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    fvgs: list[dict[str, Any]] = []

    for i in range(2, len(data)):
        left = i - 2
        atr_value = float(atr.iloc[i])
        min_gap = atr_value * DEFAULT_CONFIG.fvg_min_gap_atr

        if lows[i] > highs[left]:
            gap_low = float(highs[left])
            gap_high = float(lows[i])
            if gap_high - gap_low >= min_gap:
                fvgs.append(
                    {
                        "type": "bullish",
                        "low": gap_low,
                        "high": gap_high,
                        "center": (gap_low + gap_high) / 2.0,
                        "index": i,
                        "timestamp": resolve_timestamp(data, i),
                        "filled": False,
                    }
                )

        if highs[i] < lows[left]:
            gap_high = float(lows[left])
            gap_low = float(highs[i])
            if gap_high - gap_low >= min_gap:
                fvgs.append(
                    {
                        "type": "bearish",
                        "low": gap_low,
                        "high": gap_high,
                        "center": (gap_high + gap_low) / 2.0,
                        "index": i,
                        "timestamp": resolve_timestamp(data, i),
                        "filled": False,
                    }
                )

    return _mark_filled_fvgs(data, fvgs)


def _mark_filled_fvgs(data: pd.DataFrame, fvgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()

    for fvg in fvgs:
        start = fvg["index"] + 1
        filled = False
        for i in range(start, len(data)):
            if fvg["type"] == "bullish" and lows[i] <= fvg["low"]:
                filled = True
                fvg["fill_index"] = i
                fvg["fill_timestamp"] = resolve_timestamp(data, i)
                break
            if fvg["type"] == "bearish" and highs[i] >= fvg["high"]:
                filled = True
                fvg["fill_index"] = i
                fvg["fill_timestamp"] = resolve_timestamp(data, i)
                break
        fvg["filled"] = filled

    return fvgs


def find_fvg_retest_after_sweep(
    data: pd.DataFrame,
    sweep_bar_index: int,
    setup_type: str,
    fvgs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return FVG POI retested after sweep bar for entry alignment."""
    if sweep_bar_index < 0 or not fvgs:
        return None

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()

    for fvg in fvgs:
        if fvg["index"] > sweep_bar_index:
            continue
        if setup_type == "buy" and fvg["type"] != "bullish":
            continue
        if setup_type == "sell" and fvg["type"] != "bearish":
            continue

        gap_low = float(fvg["low"])
        gap_high = float(fvg["high"])

        for i in range(sweep_bar_index + 1, len(data)):
            if lows[i] <= gap_high and highs[i] >= gap_low:
                return fvg

    return None
