"""Institutional liquidity sweep detection with reclaim scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.confirmation import three_candle_confirmation
from liquidity_zone_engine.utils import normalize_ohlc, resolve_timestamp


def _wick_rejection(
    open_price: float,
    high: float,
    low: float,
    close: float,
    sweep_type: str,
) -> bool:
    body = abs(close - open_price) or 1e-9
    ratio = DEFAULT_CONFIG.sweep_wick_body_ratio
    if sweep_type == "buy_side_sweep":
        return (high - max(open_price, close)) >= body * ratio
    return (min(open_price, close) - low) >= body * ratio


def _closes_back_inside(
    index: int,
    sweep_type: str,
    zone_low: float,
    zone_high: float,
    closes,
    max_bars: int,
) -> bool:
    end = min(len(closes), index + max_bars + 1)
    for j in range(index, end):
        if sweep_type == "buy_side_sweep" and closes[j] < zone_high:
            return True
        if sweep_type == "sell_side_sweep" and closes[j] > zone_low:
            return True
    return False


def _has_displacement_after(
    index: int,
    close: float,
    sweep_type: str,
    closes,
    highs,
    lows,
    atr_value: float,
) -> bool:
    if index + 1 >= len(closes):
        return False
    threshold = atr_value * DEFAULT_CONFIG.sweep_min_displacement_atr
    next_close = closes[index + 1]
    if sweep_type == "buy_side_sweep":
        return next_close <= close - threshold
    return next_close >= close + threshold


def _volume_spike(index: int, volumes, window: int = 20) -> bool:
    if volumes is None or index >= len(volumes):
        return False
    start = max(0, index - window)
    segment = volumes[start:index]
    if len(segment) < 5:
        return False
    avg = sum(segment) / len(segment)
    return volumes[index] >= avg * 1.5


def _score_sweep(
    sweep_type: str,
    index: int,
    zone_level: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    closes,
    highs,
    lows,
    atr_value: float,
    volumes,
) -> int:
    config = DEFAULT_CONFIG
    score = 0
    if _wick_rejection(open_price, high, low, close, sweep_type):
        score += config.sweep_score_wick
    if _volume_spike(index, volumes):
        score += config.sweep_score_volume
    if zone_level == "macro":
        score += config.sweep_score_macro
    if _has_displacement_after(index, close, sweep_type, closes, highs, lows, atr_value):
        score += config.sweep_score_displacement
    return score


def remap_sweeps_to_zones(
    sweeps: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not zones:
        return []

    remapped: list[dict[str, Any]] = []
    for sweep in sweeps:
        old_center = sweep.get("linked_zone_center")
        if old_center is None and sweep.get("zone_index") is not None:
            idx = int(sweep["zone_index"])
            if 0 <= idx < len(zones):
                old_center = zones[idx].get("center")

        if old_center is None:
            continue

        nearest = min(
            range(len(zones)),
            key=lambda i: abs(zones[i]["center"] - float(old_center)),
        )
        item = dict(sweep)
        item["zone_index"] = nearest
        item["linked_zone_center"] = zones[nearest]["center"]
        item["linked_zone_level"] = zones[nearest].get("level")
        remapped.append(item)

    return remapped


def detect_sweeps(
    df: pd.DataFrame,
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Detect institutional sweeps: break liquidity then close back inside zone.

    Keeps only sweeps with score >= min_sweep_score (default 50).
    """
    if not zones:
        return []

    config = DEFAULT_CONFIG
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=config.atr_period)
    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()
    volumes = None
    if "tick_volume" in data.columns:
        volumes = data["tick_volume"].to_numpy()
    elif "volume" in data.columns:
        volumes = data["volume"].to_numpy()

    events: list[dict[str, Any]] = []
    last_event_bar: dict[int, int] = {}

    for zone_index, zone in enumerate(zones):
        zone_low = float(zone["low"])
        zone_high = float(zone["high"])
        zone_level = zone.get("level", "mid")

        for i in range(1, len(data)):
            if zone_index in last_event_bar and i - last_event_bar[zone_index] < 3:
                continue

            atr_value = float(atr.iloc[i])
            open_price = opens[i]
            high = highs[i]
            low = lows[i]
            close = closes[i]

            if high > zone_high:
                if _closes_back_inside(
                    i,
                    "buy_side_sweep",
                    zone_low,
                    zone_high,
                    closes,
                    config.sweep_reclaim_bars,
                ):
                    score = _score_sweep(
                        "buy_side_sweep",
                        i,
                        zone_level,
                        open_price,
                        high,
                        low,
                        close,
                        closes,
                        highs,
                        lows,
                        atr_value,
                        volumes,
                    )
                    if score >= config.min_sweep_score:
                        has_disp = _has_displacement_after(
                            i, close, "buy_side_sweep", closes, highs, lows, atr_value
                        )
                        has_wick = _wick_rejection(open_price, high, low, close, "buy_side_sweep")
                        if not (has_wick and has_disp):
                            continue
                        setup_type = "sell"
                        events.append(
                            {
                                "type": "buy_side_sweep",
                                "zone_index": zone_index,
                                "zone_level": zone_level,
                                "bar_index": i,
                                "timestamp": resolve_timestamp(data, i),
                                "price": float(close),
                                "liquidity_price": zone_high,
                                "score": score,
                                "real_sweep": True,
                                "wick_rejection": has_wick,
                                "has_displacement": has_disp,
                                "linked_zone_center": zone["center"],
                                "classification": "buy_side_liquidity_grab",
                                "three_candle_confirmed": three_candle_confirmation(
                                    data, i, setup_type
                                ),
                            }
                        )
                        last_event_bar[zone_index] = i
                        continue

            if low < zone_low:
                if _closes_back_inside(
                    i,
                    "sell_side_sweep",
                    zone_low,
                    zone_high,
                    closes,
                    config.sweep_reclaim_bars,
                ):
                    score = _score_sweep(
                        "sell_side_sweep",
                        i,
                        zone_level,
                        open_price,
                        high,
                        low,
                        close,
                        closes,
                        highs,
                        lows,
                        atr_value,
                        volumes,
                    )
                    if score >= config.min_sweep_score:
                        has_disp = _has_displacement_after(
                            i, close, "sell_side_sweep", closes, highs, lows, atr_value
                        )
                        has_wick = _wick_rejection(open_price, high, low, close, "sell_side_sweep")
                        if not (has_wick and has_disp):
                            continue
                        setup_type = "buy"
                        events.append(
                            {
                                "type": "sell_side_sweep",
                                "zone_index": zone_index,
                                "zone_level": zone_level,
                                "bar_index": i,
                                "timestamp": resolve_timestamp(data, i),
                                "price": float(close),
                                "liquidity_price": zone_low,
                                "score": score,
                                "real_sweep": True,
                                "wick_rejection": has_wick,
                                "has_displacement": has_disp,
                                "linked_zone_center": zone["center"],
                                "classification": "sell_side_liquidity_grab",
                                "three_candle_confirmed": three_candle_confirmation(
                                    data, i, setup_type
                                ),
                            }
                        )
                        last_event_bar[zone_index] = i

    events.sort(key=lambda event: (event["timestamp"], event["zone_index"]))
    return events


def _reclaim_within_bars(
    index: int,
    sweep_type: str,
    level: float,
    closes,
    min_bars: int,
    max_bars: int,
) -> bool:
    end = min(len(closes), index + max_bars + 1)
    for j in range(index + min_bars, end):
        if sweep_type == "buy_side_sweep" and closes[j] < level:
            return True
        if sweep_type == "sell_side_sweep" and closes[j] > level:
            return True
    if min_bars == 0:
        if sweep_type == "buy_side_sweep" and closes[index] < level:
            return True
        if sweep_type == "sell_side_sweep" and closes[index] > level:
            return True
    return False


def _reversal_after_sweep(
    index: int,
    sweep_type: str,
    close: float,
    closes,
) -> bool:
    if index + 1 >= len(closes):
        return False
    next_close = closes[index + 1]
    if sweep_type == "buy_side_sweep":
        return next_close < close
    return next_close > close


def _rejection_body_atr(
    index: int,
    sweep_type: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    atr_value: float,
) -> bool:
    """Rejection candle body/range must be >= 0.5 ATR."""
    threshold = atr_value * DEFAULT_CONFIG.smc_sweep_rejection_atr
    if sweep_type == "buy_side_sweep":
        rejection = high - close
    else:
        rejection = close - low
    candle_range = high - low
    return rejection >= threshold or candle_range >= threshold


def _follow_through_after_sweep(
    index: int,
    sweep_type: str,
    level: float,
    closes,
    atr_value: float,
) -> bool:
    """Next 1-2 bars must continue away from swept level."""
    if index + 1 >= len(closes):
        return False
    move = abs(closes[index + 1] - closes[index])
    if sweep_type == "buy_side_sweep":
        return closes[index + 1] < level and move >= atr_value * 0.15
    return closes[index + 1] > level and move >= atr_value * 0.15


def detect_smc_sweeps(
    df: pd.DataFrame,
    swings: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Institutional sweep: break swing level, reclaim within 3 bars,
    rejection >= 0.5 ATR, with follow-through.
    """
    config = DEFAULT_CONFIG
    data = normalize_ohlc(df)
    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()
    atr = compute_atr(data, period=config.atr_period)

    swing_highs = [s for s in swings if s["type"] == "swing_high"]
    swing_lows = [s for s in swings if s["type"] == "swing_low"]
    events: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for i in range(1, len(data)):
        atr_value = float(atr.iloc[i])

        for sh in swing_highs:
            level = float(sh["price"])
            if int(sh["index"]) >= i or highs[i] <= level:
                continue
            if not _reclaim_within_bars(i, "buy_side_sweep", level, closes, 0, config.sweep_reclaim_bars):
                continue
            if not _rejection_body_atr(i, "buy_side_sweep", opens[i], highs[i], lows[i], closes[i], atr_value):
                continue
            if not _follow_through_after_sweep(i, "buy_side_sweep", level, closes, atr_value):
                continue

            key = (i, "buy_side_sweep")
            if key in seen:
                continue
            seen.add(key)

            score = config.smc_sweep_base_score
            if _reversal_after_sweep(i, "buy_side_sweep", closes[i], closes):
                score += config.smc_sweep_reversal_score

            zone_index = None
            if zones:
                zone_index = min(range(len(zones)), key=lambda z: abs(zones[z]["center"] - level))
                if abs(zones[zone_index]["center"] - level) / max(level, 1e-9) <= 0.002:
                    score += config.smc_sweep_zone_score

            events.append(
                {
                    "type": "buy_side_sweep",
                    "bar_index": i,
                    "timestamp": resolve_timestamp(data, i),
                    "price": round(float(closes[i]), 5),
                    "liquidity_price": round(level, 5),
                    "score": score,
                    "confirmed": True,
                    "real_sweep": True,
                    "zone_index": zone_index,
                    "classification": "buy_side_liquidity_grab",
                }
            )

        for sl in swing_lows:
            level = float(sl["price"])
            if int(sl["index"]) >= i or lows[i] >= level:
                continue
            if not _reclaim_within_bars(i, "sell_side_sweep", level, closes, 0, config.sweep_reclaim_bars):
                continue
            if not _rejection_body_atr(i, "sell_side_sweep", opens[i], highs[i], lows[i], closes[i], atr_value):
                continue
            if not _follow_through_after_sweep(i, "sell_side_sweep", level, closes, atr_value):
                continue

            key = (i, "sell_side_sweep")
            if key in seen:
                continue
            seen.add(key)

            score = config.smc_sweep_base_score
            if _reversal_after_sweep(i, "sell_side_sweep", closes[i], closes):
                score += config.smc_sweep_reversal_score

            zone_index = None
            if zones:
                zone_index = min(range(len(zones)), key=lambda z: abs(zones[z]["center"] - level))
                if abs(zones[zone_index]["center"] - level) / max(level, 1e-9) <= 0.002:
                    score += config.smc_sweep_zone_score

            events.append(
                {
                    "type": "sell_side_sweep",
                    "bar_index": i,
                    "timestamp": resolve_timestamp(data, i),
                    "price": round(float(closes[i]), 5),
                    "liquidity_price": round(level, 5),
                    "score": score,
                    "confirmed": True,
                    "real_sweep": True,
                    "zone_index": zone_index,
                    "classification": "sell_side_liquidity_grab",
                }
            )

    events.sort(key=lambda e: (e["timestamp"], e["bar_index"]))
    return events
