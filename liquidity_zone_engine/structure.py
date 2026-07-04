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


def _detect_bos_events(
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
    atr: pd.Series,
) -> list[dict[str, Any]]:
    """Detect real vs fake breaks of structure (inducement)."""
    config = DEFAULT_CONFIG
    closes = data["close"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    events: list[dict[str, Any]] = []

    for i in range(1, len(swings)):
        swing = swings[i]
        prior = swings[i - 1]
        idx = swing["index"]
        atr_value = float(atr.iloc[idx])
        buffer = atr_value * config.bos_confirm_atr

        if swing["type"] == "swing_high" and prior["type"] == "swing_low":
            level = prior["price"]
            segment = closes[prior["index"] : idx + 1]
            if segment.max() <= level + buffer:
                continue

            retrace_window = closes[idx + 1 : idx + 1 + config.bos_fake_retrace_bars]
            fake = len(retrace_window) > 0 and retrace_window.min() < level
            wick_only = highs[idx] > level + buffer and closes[idx] < level + buffer

            events.append(
                {
                    "index": idx,
                    "direction": "bullish",
                    "bos_type": "fake" if fake or wick_only else "real",
                    "level": level,
                    "timestamp": swing["timestamp"],
                }
            )

        if swing["type"] == "swing_low" and prior["type"] == "swing_high":
            level = prior["price"]
            segment = closes[prior["index"] : idx + 1]
            if segment.min() >= level - buffer:
                continue

            retrace_window = closes[idx + 1 : idx + 1 + config.bos_fake_retrace_bars]
            fake = len(retrace_window) > 0 and retrace_window.max() > level
            wick_only = lows[idx] < level - buffer and closes[idx] > level - buffer

            events.append(
                {
                    "index": idx,
                    "direction": "bearish",
                    "bos_type": "fake" if fake or wick_only else "real",
                    "level": level,
                    "timestamp": swing["timestamp"],
                }
            )

    return events


def _annotate_idm_and_structure(
    swings: list[dict[str, Any]],
    bos_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mark first internal swing after BOS as IDM (inducement trap)."""
    annotated: list[dict[str, Any]] = []
    idm_marked: set[int] = set()

    for event in bos_events:
        event_index = event["index"]
        internal: dict[str, Any] | None = None
        for swing in swings:
            if swing["index"] <= event_index:
                continue
            if event["direction"] == "bullish" and swing["type"] == "swing_low":
                internal = swing
                break
            if event["direction"] == "bearish" and swing["type"] == "swing_high":
                internal = swing
                break
        if internal is not None:
            idm_marked.add(internal["index"])

    for swing in swings:
        item = dict(swing)
        item["is_idm"] = swing["index"] in idm_marked
        item["structure_role"] = "idm" if item["is_idm"] else "structural"

        related_bos = next((e for e in bos_events if e["index"] == swing["index"]), None)
        if related_bos:
            item["bos_type"] = related_bos["bos_type"]
        else:
            item["bos_type"] = None

        annotated.append(item)

    return annotated


def annotate_liquidity_roles(
    swings: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label protected (liquidity taken) vs targeted swing levels."""
    swept_prices: list[float] = []
    for sweep in sweeps:
        zone_index = sweep.get("zone_index", -1)
        if zone_index < 0 or zone_index >= len(zones):
            continue
        zone = zones[zone_index]
        if sweep["type"] == "sell_side_sweep":
            swept_prices.append(float(zone["low"]))
        else:
            swept_prices.append(float(zone["high"]))

    updated: list[dict[str, Any]] = []
    for swing in swings:
        item = dict(swing)
        price = float(swing["price"])
        tolerance = abs(price) * 0.0003 + 0.05

        protected = any(abs(price - level) <= tolerance for level in swept_prices)
        if protected:
            item["liquidity_status"] = "protected"
        elif swing.get("is_idm"):
            item["liquidity_status"] = "idm_trap"
        else:
            item["liquidity_status"] = "targeted"

        updated.append(item)

    return updated


def has_break_of_structure(
    data: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    swings: list[dict[str, Any]],
    atr: pd.Series | None = None,
) -> bool:
    """Return True if price confirmed a REAL structure break between indices."""
    start, end = sorted((idx_a, idx_b))
    if start == end:
        return False

    closes = data["close"].to_numpy()[start : end + 1]
    prior = [s for s in swings if s["index"] < start]
    if len(prior) == 0 or len(closes) == 0:
        return False

    if atr is not None:
        buffer = float(atr.iloc[start:end + 1].mean()) * DEFAULT_CONFIG.bos_confirm_atr
    else:
        buffer = 0.0

    prior_highs = [s for s in prior if s["type"] == "swing_high" and s.get("bos_type") != "fake"]
    prior_lows = [s for s in prior if s["type"] == "swing_low" and s.get("bos_type") != "fake"]

    if prior_highs:
        last_high = prior_highs[-1]["price"]
        if closes.max() > last_high + buffer:
            return True

    if prior_lows:
        last_low = prior_lows[-1]["price"]
        if closes.min() < last_low - buffer:
            return True

    return False


def _filter_lit_structure_swings(
    swings: list[dict[str, Any]],
    data: pd.DataFrame,
    atr: pd.Series,
) -> list[dict[str, Any]]:
    """Keep swings that break prior structure and show post-swing displacement."""
    config = DEFAULT_CONFIG
    closes = data["close"].to_numpy()
    kept: list[dict[str, Any]] = []

    for swing in swings:
        idx = int(swing["index"])
        atr_value = float(atr.iloc[idx])
        threshold = atr_value * config.lit_swing_post_displacement_atr

        if swing["type"] == "swing_high":
            prior_highs = [s for s in kept if s["type"] == "swing_high"]
            if prior_highs and float(swing["price"]) <= float(prior_highs[-1]["price"]):
                continue
            forward = closes[idx + 1 : idx + 4]
            if len(forward) == 0 or (float(swing["price"]) - float(min(forward))) < threshold:
                continue
        else:
            prior_lows = [s for s in kept if s["type"] == "swing_low"]
            if prior_lows and float(swing["price"]) >= float(prior_lows[-1]["price"]):
                continue
            forward = closes[idx + 1 : idx + 4]
            if len(forward) == 0 or (float(max(forward)) - float(swing["price"])) < threshold:
                continue

        kept.append(swing)

    return kept


def detect_swings(df: pd.DataFrame, swing_window: int = 3) -> list[dict[str, Any]]:
    """
    Controlled swing extraction: fractal ±3, significant distance filter, max 20.
    """
    if swing_window < 1:
        raise ValueError("swing_window must be >= 1")

    from liquidity_zone_engine.smc_dashboard import filter_controlled_swings

    config = DEFAULT_CONFIG
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=config.atr_period)
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()

    raw = _find_raw_swings(data, highs, lows, swing_window)
    filtered = filter_controlled_swings(raw, data, atr)

    return [
        {
            "type": swing["type"],
            "index": swing["index"],
            "timestamp": swing["timestamp"],
            "price": swing["price"],
            "leg_type": "impulse",
            "is_idm": False,
            "structure_role": "structural",
            "bos_type": None,
            "liquidity_status": "targeted",
        }
        for swing in filtered
    ]
