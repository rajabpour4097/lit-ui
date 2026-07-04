"""Liquidity sweep detection (LIT logic)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc, resolve_timestamp


def _is_rejection_candle(
    open_price: float,
    high: float,
    low: float,
    close: float,
    sweep_type: str,
) -> bool:
    body = abs(close - open_price) or 1e-9

    if sweep_type == "buy_side_sweep":
        upper_wick = high - max(open_price, close)
        return upper_wick >= body * 0.8

    lower_wick = min(open_price, close) - low
    return lower_wick >= body * 0.8


def _has_displacement_after_sweep(
    index: int,
    close: float,
    sweep_type: str,
    closes,
    highs,
    lows,
    atr_value: float,
) -> bool:
    """Displacement impulse away from the zone must occur after the sweep bar."""
    if index + 1 >= len(closes):
        return False

    threshold = atr_value * DEFAULT_CONFIG.sweep_displacement_atr
    next_close = closes[index + 1]

    if sweep_type == "buy_side_sweep":
        bearish_impulse = next_close <= close - threshold
        next_extension = highs[index + 1] <= highs[index]
        return bearish_impulse and next_extension

    bullish_impulse = next_close >= close + threshold
    next_extension = lows[index + 1] >= lows[index]
    return bullish_impulse and next_extension


def _was_inside_zone(prev_close: float, zone_low: float, zone_high: float) -> bool:
    return zone_low <= prev_close <= zone_high


def detect_sweeps(
    df: pd.DataFrame,
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Detect liquidity sweeps against pre-built zones.

    Valid sweep requires ALL of:
    - wick breaks zone boundary
    - close back inside zone OR rejection candle
    - displacement impulse on the following bar (mandatory)
    """
    if not zones:
        return []

    data = normalize_ohlc(df)
    atr = compute_atr(data, period=DEFAULT_CONFIG.atr_period)
    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()

    events: list[dict[str, Any]] = []
    last_event_bar: dict[int, int] = {}

    for zone_index, zone in enumerate(zones):
        zone_low = float(zone["low"])
        zone_high = float(zone["high"])

        for i in range(1, len(data)):
            if zone_index in last_event_bar and i - last_event_bar[zone_index] < 5:
                continue

            prev_close = closes[i - 1]
            if not _was_inside_zone(prev_close, zone_low, zone_high):
                continue

            atr_value = float(atr.iloc[i])
            open_price = opens[i]
            high = highs[i]
            low = lows[i]
            close = closes[i]

            buy_wick_beyond = high > zone_high
            buy_close_inside = zone_low <= close <= zone_high
            buy_rejection = _is_rejection_candle(
                open_price, high, low, close, "buy_side_sweep"
            )

            if buy_wick_beyond and (buy_close_inside or buy_rejection):
                if _has_displacement_after_sweep(
                    i,
                    close,
                    "buy_side_sweep",
                    closes,
                    highs,
                    lows,
                    atr_value,
                ):
                    events.append(
                        {
                            "type": "buy_side_sweep",
                            "zone_index": zone_index,
                            "timestamp": resolve_timestamp(data, i),
                            "price": float(close),
                        }
                    )
                    last_event_bar[zone_index] = i
                    continue

            sell_wick_beyond = low < zone_low
            sell_close_inside = zone_low <= close <= zone_high
            sell_rejection = _is_rejection_candle(
                open_price, high, low, close, "sell_side_sweep"
            )

            if sell_wick_beyond and (sell_close_inside or sell_rejection):
                if _has_displacement_after_sweep(
                    i,
                    close,
                    "sell_side_sweep",
                    closes,
                    highs,
                    lows,
                    atr_value,
                ):
                    events.append(
                        {
                            "type": "sell_side_sweep",
                            "zone_index": zone_index,
                            "timestamp": resolve_timestamp(data, i),
                            "price": float(close),
                        }
                    )
                    last_event_bar[zone_index] = i

    events.sort(key=lambda event: (event["timestamp"], event["zone_index"]))
    return events
