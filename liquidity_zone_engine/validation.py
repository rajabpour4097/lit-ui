"""SMC validation, cleaning, and dashboard-ready output for liquidity analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc
from liquidity_zone_engine.zones import flatten_zone_map, regroup_zones


def _zone_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap_low = max(left["low"], right["low"])
    overlap_high = min(left["high"], right["high"])
    if overlap_high <= overlap_low:
        return 0.0
    overlap = overlap_high - overlap_low
    smaller = min(left["high"] - left["low"], right["high"] - right["low"])
    if smaller <= 0:
        return 0.0
    return overlap / smaller


def _nearest_swing_distance(zone: dict[str, Any], swings: list[dict[str, Any]]) -> float:
    if not swings:
        return float("inf")
    return min(abs(float(swing["price"]) - zone["center"]) for swing in swings)


def _classify_zone_type(zone: dict[str, Any], last_close: float) -> str:
    if last_close > zone["high"]:
        return "Low"
    if last_close < zone["low"]:
        return "High"
    mid = (zone["high"] + zone["low"]) / 2.0
    zone_height = zone["high"] - zone["low"]
    if zone_height <= 0:
        return "Equilibrium"
    if abs(last_close - mid) <= zone_height * 0.35:
        return "Equilibrium"
    return "High" if last_close < mid else "Low"


def _merge_nearby_zones(
    zones: list[dict[str, Any]],
    atr_value: float,
) -> tuple[list[dict[str, Any]], int]:
    if not zones:
        return [], 0

    config = DEFAULT_CONFIG
    threshold = atr_value * config.validation_merge_atr
    merged_count = 0
    ordered = sorted(zones, key=lambda item: (item.get("level_rank", 2), item["center"]))
    result: list[dict[str, Any]] = []

    for zone in ordered:
        merged = False
        for i, existing in enumerate(result):
            same_level = existing.get("level") == zone.get("level")
            center_gap = abs(existing["center"] - zone["center"])
            overlap = _zone_overlap_ratio(existing, zone)
            if same_level and (center_gap <= threshold or overlap >= 0.4):
                existing["low"] = min(existing["low"], zone["low"])
                existing["high"] = max(existing["high"], zone["high"])
                existing["center"] = (existing["low"] + existing["high"]) / 2.0
                existing["touch_count"] = int(existing.get("touch_count", 0)) + int(
                    zone.get("touch_count", 0)
                )
                existing["sweep_count"] = int(existing.get("sweep_count", 0)) + int(
                    zone.get("sweep_count", 0)
                )
                existing["cluster_point_count"] = int(
                    existing.get("cluster_point_count", 1)
                ) + int(zone.get("cluster_point_count", 1))
                existing["strength"] = max(int(existing.get("strength", 0)), int(zone.get("strength", 0)))
                merged = True
                merged_count += 1
                break
        if not merged:
            result.append(dict(zone))

    return result, merged_count


def _recalibrate_strength(
    zone: dict[str, Any],
    swings: list[dict[str, Any]],
    atr_value: float,
) -> int:
    config = DEFAULT_CONFIG
    touch_score = min(zone.get("touch_count", 0), 6) * 8
    sweep_score = min(zone.get("sweep_count", 0), 4) * 12
    swing_distance = _nearest_swing_distance(zone, swings)
    swing_score = 0.0
    if swing_distance <= atr_value * config.swing_alignment_atr:
        swing_score = 25.0
    elif swing_distance <= atr_value:
        swing_score = 12.0

    reaction_score = min(zone.get("cluster_point_count", 1), 5) * 4
    level_bonus = {"macro": 15, "mid": 8, "micro": 0}.get(zone.get("level", "mid"), 0)
    raw = touch_score + sweep_score + swing_score + reaction_score + level_bonus
    return int(min(100, max(0, round(raw))))


def _trend_from_swings(swings: list[dict[str, Any]]) -> str:
    if len(swings) < 4:
        return "range"

    recent = sorted(swings, key=lambda item: item["index"])[-8:]
    highs = [item for item in recent if item["type"] == "swing_high"]
    lows = [item for item in recent if item["type"] == "swing_low"]

    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1]["price"] > highs[-2]["price"]
        hl = lows[-1]["price"] > lows[-2]["price"]
        lh = highs[-1]["price"] < highs[-2]["price"]
        ll = lows[-1]["price"] < lows[-2]["price"]
        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"

    return "range"


def _key_bos_levels(swings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for swing in swings:
        if swing.get("bos_type") != "real":
            continue
        levels.append(
            {
                "direction": "bullish" if swing["type"] == "swing_high" else "bearish",
                "level": float(swing["price"]),
                "timestamp": swing.get("timestamp"),
            }
        )
    return levels[-4:]


def _liquidity_targets(
    zones: list[dict[str, Any]],
    last_close: float,
) -> dict[str, Any]:
    above = sorted(
        [zone for zone in zones if zone["center"] > last_close],
        key=lambda item: item["center"],
    )
    below = sorted(
        [zone for zone in zones if zone["center"] < last_close],
        key=lambda item: item["center"],
        reverse=True,
    )

    return {
        "buy_side_liquidity": {
            "center": above[0]["center"],
            "strength": above[0]["strength"],
            "level": above[0].get("level"),
        }
        if above
        else None,
        "sell_side_liquidity": {
            "center": below[0]["center"],
            "strength": below[0]["strength"],
            "level": below[0].get("level"),
        }
        if below
        else None,
    }


def _validate_sweeps(
    sweeps: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    swings: list[dict[str, Any]],
    atr_value: float,
) -> tuple[list[dict[str, Any]], int]:
    config = DEFAULT_CONFIG
    validated: list[dict[str, Any]] = []
    removed = 0

    for sweep in sweeps:
        if not sweep.get("real_sweep", True):
            removed += 1
            continue

        zone_index = sweep.get("zone_index")
        if zone_index is None or zone_index >= len(zones):
            removed += 1
            continue

        zone = zones[zone_index]
        bar_index = int(sweep.get("bar_index", -1))
        nearby_swing = False
        for swing in swings:
            if abs(int(swing["index"]) - bar_index) > 20:
                continue
            if abs(float(swing["price"]) - zone["center"]) <= atr_value * config.swing_alignment_atr:
                nearby_swing = True
                break

        if not nearby_swing and zone.get("level") != "macro":
            removed += 1
            continue

        item = dict(sweep)
        item["linked_zone_center"] = zone["center"]
        item["linked_zone_level"] = zone.get("level")
        item["classification"] = (
            "buy_side_liquidity_grab"
            if sweep["type"] == "buy_side_sweep"
            else "sell_side_liquidity_grab"
        )
        validated.append(item)

    return validated, removed


def validate_dashboard_output(
    df: pd.DataFrame,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Review, clean, and enrich liquidity analysis for dashboard display.

    Merges redundant zones, removes noise, recalibrates strength, validates sweeps,
    and produces market-structure context with a fix summary.
    """
    config = DEFAULT_CONFIG
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=config.atr_period)
    atr_value = float(atr.iloc[-1]) if len(atr) else 0.0
    last_close = float(data["close"].iloc[-1])

    swings = analysis.get("swings", [])
    raw_zones = flatten_zone_map(analysis.get("zones", {}))
    raw_sweeps = list(analysis.get("sweeps", []))

    zones_before = len(raw_zones)
    sweeps_before = len(raw_sweeps)

    merged_zones, merged_count = _merge_nearby_zones(raw_zones, atr_value)

    kept_zones: list[dict[str, Any]] = []
    removed_weak = 0
    removed_orphan = 0

    for zone in merged_zones:
        swing_distance = _nearest_swing_distance(zone, swings)
        structurally_important = zone.get("level") == "macro" or zone.get("sweep_count", 0) >= 2

        if swing_distance > atr_value * config.swing_alignment_atr * 2 and not structurally_important:
            removed_orphan += 1
            continue

        strength = _recalibrate_strength(zone, swings, atr_value)
        zone = dict(zone)
        zone["strength"] = strength
        zone["zone_type"] = _classify_zone_type(zone, last_close)
        zone["liquidity_class"] = (
            "external" if zone.get("level") == "macro" else "internal"
        )

        if strength < config.min_institutional_strength and not structurally_important:
            removed_weak += 1
            continue

        kept_zones.append(zone)

    kept_zones.sort(key=lambda item: item["center"], reverse=True)
    validated_sweeps, removed_sweeps = _validate_sweeps(
        raw_sweeps,
        kept_zones,
        swings,
        atr_value,
    )

    zones_grouped = regroup_zones(kept_zones)
    trend = _trend_from_swings(swings)
    bos_levels = _key_bos_levels(swings)
    liquidity_targets = _liquidity_targets(kept_zones, last_close)

    fix_summary = {
        "zones_before": zones_before,
        "zones_after": len(kept_zones),
        "zones_merged": merged_count,
        "zones_removed_weak": removed_weak,
        "zones_removed_orphan": removed_orphan,
        "sweeps_before": sweeps_before,
        "sweeps_after": len(validated_sweeps),
        "sweeps_removed": removed_sweeps,
        "notes": [],
    }

    if merged_count:
        fix_summary["notes"].append(
            f"{merged_count} زون همپوشان در فاصله {config.validation_merge_atr}×ATR ادغام شد."
        )
    if removed_weak:
        fix_summary["notes"].append(
            f"{removed_weak} زون ضعیف (قدرت < {config.min_institutional_strength}) حذف شد."
        )
    if removed_orphan:
        fix_summary["notes"].append(
            f"{removed_orphan} زون بدون سویینگ ساختاری در فضای خالی قیمت حذف شد."
        )
    if removed_sweeps:
        fix_summary["notes"].append(
            f"{removed_sweeps} sweep نویز یا بدون زون معتبر حذف شد."
        )
    if not fix_summary["notes"]:
        fix_summary["notes"].append("خروجی با ساختار بازار همخوان بود؛ فقط strength بازکالیبره شد.")

    return {
        "zones": zones_grouped,
        "zones_flat": kept_zones,
        "sweeps": validated_sweeps,
        "market_structure": {
            "trend": trend,
            "last_close": round(last_close, 5),
            "key_bos_levels": bos_levels,
            "liquidity_targets": liquidity_targets,
            "zone_count_by_level": {
                level: len(zones_grouped.get(level, []))
                for level in ("macro", "mid", "micro")
            },
        },
        "fix_summary": fix_summary,
    }
