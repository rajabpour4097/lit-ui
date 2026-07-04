"""BOS / CHOCH detection — strict anti-noise institutional rules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc, resolve_timestamp


def _has_displacement_after(
    data: pd.DataFrame,
    index: int,
    direction: str,
    atr: pd.Series,
) -> bool:
    closes = data["close"].to_numpy()
    if index + 1 >= len(closes):
        return False
    threshold = float(atr.iloc[index]) * DEFAULT_CONFIG.sweep_displacement_atr
    if direction == "bullish":
        return closes[index + 1] >= closes[index] + threshold
    return closes[index + 1] <= closes[index] - threshold


def _not_reverted_within(
    data: pd.DataFrame,
    index: int,
    level: float,
    direction: str,
) -> bool:
    closes = data["close"].to_numpy()
    end = min(len(closes), index + 1 + DEFAULT_CONFIG.bos_revert_bars)
    for j in range(index + 1, end):
        if direction == "bullish" and closes[j] < level:
            return False
        if direction == "bearish" and closes[j] > level:
            return False
    return True


def _is_valid_bos(
    data: pd.DataFrame,
    swing: dict[str, Any],
    prior: dict[str, Any],
    atr: pd.Series,
) -> bool:
    config = DEFAULT_CONFIG
    idx = int(swing["index"])
    closes = data["close"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    atr_value = float(atr.iloc[idx])
    min_break = atr_value * config.bos_break_atr

    if swing["type"] == "swing_high" and prior["type"] == "swing_low":
        level = float(prior["price"])
        if closes[idx] <= level:
            return False
        if closes[idx] - level < min_break:
            return False
        if highs[idx] > level and closes[idx] <= level:
            return False
        return _not_reverted_within(data, idx, level, "bullish")

    if swing["type"] == "swing_low" and prior["type"] == "swing_high":
        level = float(prior["price"])
        if closes[idx] >= level:
            return False
        if level - closes[idx] < min_break:
            return False
        if lows[idx] < level and closes[idx] >= level:
            return False
        return _not_reverted_within(data, idx, level, "bearish")

    return False


def detect_bos_choch_timeline(
    df: pd.DataFrame,
    swings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build filtered BOS / CHOCH timeline (structure breaks only)."""
    if len(swings) < 2:
        return []

    data = normalize_ohlc(df)
    atr = compute_atr(data, period=DEFAULT_CONFIG.atr_period)
    timeline: list[dict[str, Any]] = []
    trend_direction: str | None = None

    for i in range(1, len(swings)):
        swing = swings[i]
        prior = swings[i - 1]
        if not _is_valid_bos(data, swing, prior, atr):
            continue

        idx = int(swing["index"])
        direction = "bullish" if swing["type"] == "swing_high" else "bearish"
        level = float(prior["price"])

        event_type = "BOS"
        if trend_direction is None:
            trend_direction = direction
        elif direction != trend_direction:
            if not _has_displacement_after(data, idx, direction, atr):
                continue
            event_type = "CHOCH"
            trend_direction = direction

        timeline.append(
            {
                "type": event_type,
                "direction": direction,
                "level": round(level, 5),
                "validity": "confirmed",
                "index": idx,
                "timestamp": resolve_timestamp(data, idx),
            }
        )

    return timeline
