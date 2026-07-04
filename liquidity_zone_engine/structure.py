"""Market structure detection: swing highs and swing lows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc, resolve_timestamp


def _find_raw_swings(
    data: pd.DataFrame,
    highs,
    lows,
    swing_window: int,
) -> list[dict[str, Any]]:
    swings: list[dict[str, Any]] = []
    n = len(data)

    for i in range(swing_window, n - swing_window):
        left = i - swing_window
        right = i + swing_window + 1
        window_highs = highs[left:right]
        window_lows = lows[left:right]

        if highs[i] == window_highs.max() and (window_highs > highs[i]).sum() == 0:
            if (window_highs == highs[i]).sum() == 1:
                swings.append(
                    {
                        "type": "swing_high",
                        "index": i,
                        "timestamp": resolve_timestamp(data, i),
                        "price": float(highs[i]),
                    }
                )

        if lows[i] == window_lows.min() and (window_lows < lows[i]).sum() == 0:
            if (window_lows == lows[i]).sum() == 1:
                swings.append(
                    {
                        "type": "swing_low",
                        "index": i,
                        "timestamp": resolve_timestamp(data, i),
                        "price": float(lows[i]),
                    }
                )

    swings.sort(key=lambda item: item["index"])
    return swings


def _filter_by_displacement(
    swings: list[dict[str, Any]],
    atr: pd.Series,
    min_mult: float,
) -> list[dict[str, Any]]:
    if not swings:
        return []

    kept: list[dict[str, Any]] = []
    last_high: dict[str, Any] | None = None
    last_low: dict[str, Any] | None = None

    for swing in swings:
        idx = swing["index"]
        atr_value = float(atr.iloc[idx])
        min_move = atr_value * min_mult

        if swing["type"] == "swing_high":
            reference = last_low["price"] if last_low else None
            if reference is not None and (swing["price"] - reference) < min_move:
                continue
            last_high = swing
        else:
            reference = last_high["price"] if last_high else None
            if reference is not None and (reference - swing["price"]) < min_move:
                continue
            last_low = swing

        kept.append(swing)

    return kept


def _merge_consolidation_swings(
    swings: list[dict[str, Any]],
    atr: pd.Series,
    consolidation_mult: float,
) -> list[dict[str, Any]]:
    if not swings:
        return []

    merged: list[dict[str, Any]] = [dict(swings[0])]

    for swing in swings[1:]:
        prev = merged[-1]
        if swing["type"] != prev["type"]:
            merged.append(dict(swing))
            continue

        atr_value = float(atr.iloc[swing["index"]])
        threshold = atr_value * consolidation_mult
        price_gap = abs(swing["price"] - prev["price"])

        if price_gap <= threshold:
            if swing["type"] == "swing_high" and swing["price"] >= prev["price"]:
                merged[-1] = dict(swing)
            elif swing["type"] == "swing_low" and swing["price"] <= prev["price"]:
                merged[-1] = dict(swing)
        else:
            merged.append(dict(swing))

    return merged


def _classify_swing_legs(
    swings: list[dict[str, Any]],
    atr: pd.Series,
    impulse_mult: float,
) -> list[dict[str, Any]]:
    if not swings:
        return []

    classified: list[dict[str, Any]] = []
    prev_swing: dict[str, Any] | None = None

    for swing in swings:
        item = dict(swing)
        idx = item["index"]
        atr_value = float(atr.iloc[idx])

        if prev_swing is None:
            item["leg_type"] = "corrective"
        else:
            displacement = abs(item["price"] - prev_swing["price"])
            item["leg_type"] = (
                "impulsive" if displacement >= atr_value * impulse_mult else "corrective"
            )

        classified.append(item)
        prev_swing = item

    return classified


def has_break_of_structure(
    data: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    swings: list[dict[str, Any]],
    atr: pd.Series | None = None,
) -> bool:
    """Return True if price confirmed a structure break between two swing indices."""
    start, end = sorted((idx_a, idx_b))
    if start == end:
        return False

    closes = data["close"].to_numpy()[start : end + 1]
    prior = [s for s in swings if s["index"] < start]
    if len(prior) == 0 or len(closes) == 0:
        return False

    if atr is not None:
        buffer = float(atr.iloc[start:end + 1].mean()) * 0.25
    else:
        buffer = 0.0

    prior_highs = [s["price"] for s in prior if s["type"] == "swing_high"]
    prior_lows = [s["price"] for s in prior if s["type"] == "swing_low"]

    if prior_highs:
        last_high = prior_highs[-1]
        if closes.max() > last_high + buffer:
            return True

    if prior_lows:
        last_low = prior_lows[-1]
        if closes.min() < last_low - buffer:
            return True

    return False


def detect_swings(df: pd.DataFrame, swing_window: int = 3) -> list[dict[str, Any]]:
    """
    Detect swing highs and swing lows using a symmetric lookback/lookahead window.

    Applies ATR displacement filtering and consolidation merge to reduce noise.
    """
    if swing_window < 1:
        raise ValueError("swing_window must be >= 1")

    config = DEFAULT_CONFIG
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=config.atr_period)
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()

    raw = _find_raw_swings(data, highs, lows, swing_window)
    filtered = _filter_by_displacement(
        raw,
        atr,
        min_mult=config.min_swing_displacement_atr,
    )
    consolidated = _merge_consolidation_swings(
        filtered,
        atr,
        consolidation_mult=config.consolidation_atr,
    )
    classified = _classify_swing_legs(
        consolidated,
        atr,
        impulse_mult=config.impulse_displacement_atr,
    )

    deduped: list[dict[str, Any]] = []
    for swing in classified:
        if not deduped:
            deduped.append(swing)
            continue

        prev = deduped[-1]
        if swing["type"] == prev["type"]:
            atr_value = float(atr.iloc[swing["index"]])
            if abs(swing["price"] - prev["price"]) <= atr_value * config.consolidation_atr:
                if swing["type"] == "swing_high" and swing["price"] > prev["price"]:
                    deduped[-1] = swing
                elif swing["type"] == "swing_low" and swing["price"] < prev["price"]:
                    deduped[-1] = swing
                continue

        deduped.append(swing)

    return [
        {
            "type": swing["type"],
            "index": swing["index"],
            "timestamp": swing["timestamp"],
            "price": swing["price"],
            "leg_type": swing["leg_type"],
        }
        for swing in deduped
    ]
