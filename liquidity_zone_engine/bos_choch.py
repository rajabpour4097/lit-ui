"""BOS / CHOCH — strict structure model (last 5 events for regime)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.utils import normalize_ohlc, resolve_timestamp


def detect_bos_choch_timeline(
    df: pd.DataFrame,
    swings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    BOS up = close breaks last valid swing high.
    BOS down = close breaks last valid swing low.
    CHOCH = first opposite BOS after trend established.
    """
    if len(swings) < 2:
        return []

    data = normalize_ohlc(df)
    closes = data["close"].to_numpy()
    highs_arr = data["high"].to_numpy()
    lows_arr = data["low"].to_numpy()
    timeline: list[dict[str, Any]] = []
    trend_direction: str | None = None

    swing_highs = [s for s in swings if s["type"] == "swing_high"]
    swing_lows = [s for s in swings if s["type"] == "swing_low"]

    last_valid_high: float | None = None
    last_valid_low: float | None = None
    broken_high_levels: set[float] = set()
    broken_low_levels: set[float] = set()

    hi_ptr = 0
    lo_ptr = 0

    for i in range(len(data)):
        while hi_ptr < len(swing_highs) and int(swing_highs[hi_ptr]["index"]) <= i:
            last_valid_high = float(swing_highs[hi_ptr]["price"])
            hi_ptr += 1
        while lo_ptr < len(swing_lows) and int(swing_lows[lo_ptr]["index"]) <= i:
            last_valid_low = float(swing_lows[lo_ptr]["price"])
            lo_ptr += 1

        if last_valid_high is not None and last_valid_high not in broken_high_levels:
            if closes[i] > last_valid_high or highs_arr[i] > last_valid_high:
                if closes[i] > last_valid_high:
                    direction = "bullish"
                    event_type = "BOS"
                    if trend_direction is None:
                        trend_direction = direction
                    elif direction != trend_direction:
                        event_type = "CHOCH"
                        trend_direction = direction

                    timeline.append(
                        {
                            "type": event_type,
                            "direction": direction,
                            "level": round(last_valid_high, 5),
                            "validity": "confirmed",
                            "index": i,
                            "timestamp": resolve_timestamp(data, i),
                        }
                    )
                    broken_high_levels.add(last_valid_high)

        if last_valid_low is not None and last_valid_low not in broken_low_levels:
            if closes[i] < last_valid_low or lows_arr[i] < last_valid_low:
                if closes[i] < last_valid_low:
                    direction = "bearish"
                    event_type = "BOS"
                    if trend_direction is None:
                        trend_direction = direction
                    elif direction != trend_direction:
                        event_type = "CHOCH"
                        trend_direction = direction

                    timeline.append(
                        {
                            "type": event_type,
                            "direction": direction,
                            "level": round(last_valid_low, 5),
                            "validity": "confirmed",
                            "index": i,
                            "timestamp": resolve_timestamp(data, i),
                        }
                    )
                    broken_low_levels.add(last_valid_low)

    seen: set[tuple[int, str, float]] = set()
    deduped: list[dict[str, Any]] = []
    for event in timeline:
        key = (int(event["index"]), event["direction"], float(event["level"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)

    return deduped
