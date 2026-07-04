"""Institutional 3-layer liquidity zone model (macro / mid / micro)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.config import DEFAULT_CONFIG, ZONE_LEVELS
from liquidity_zone_engine.zones import (
    _assign_nesting,
    _count_touch_events,
    _count_zone_interactions,
    _zone_overlap_ratio,
    flatten_zone_map,
    regroup_zones,
)


def _merge_by_overlap(
    zones: list[dict[str, Any]],
    min_overlap: float,
) -> list[dict[str, Any]]:
    if not zones:
        return []

    ordered = sorted(zones, key=lambda z: z.get("strength", 0), reverse=True)
    merged: list[dict[str, Any]] = []

    for zone in ordered:
        absorbed = False
        for existing in merged:
            if _zone_overlap_ratio(existing, zone) >= min_overlap:
                existing["low"] = min(existing["low"], zone["low"])
                existing["high"] = max(existing["high"], zone["high"])
                existing["mid"] = (existing["low"] + existing["high"]) / 2.0
                existing["center"] = existing["mid"]
                existing["strength"] = max(existing.get("strength", 0), zone.get("strength", 0))
                existing["touch_count"] = int(existing.get("touch_count", 0)) + int(
                    zone.get("touch_count", 0)
                )
                existing["sweep_count"] = int(existing.get("sweep_count", 0)) + int(
                    zone.get("sweep_count", 0)
                )
                absorbed = True
                break
        if not absorbed:
            item = dict(zone)
            item["mid"] = (item["low"] + item["high"]) / 2.0
            merged.append(item)

    return merged


def _macro_zone_type(zone: dict[str, Any], data: pd.DataFrame) -> str:
    closes = data["close"].to_numpy()
    segment = closes[int(max(0, len(closes) - 40)) :]
    if len(segment) < 5:
        return "equilibrium"

    mid = zone["mid"]
    in_zone = sum(1 for c in segment if zone["low"] <= c <= zone["high"])
    ratio = in_zone / len(segment)
    drift = segment[-1] - segment[0]

    if ratio >= 0.55 and abs(drift) < (zone["high"] - zone["low"]) * 0.3:
        return "equilibrium"
    if drift > 0:
        return "expansion"
    return "distribution"


def _layer_strength(
    zone: dict[str, Any],
    level: str,
    data: pd.DataFrame,
    atr: pd.Series,
) -> int:
    config = DEFAULT_CONFIG
    touch_count, rejection_count, displacement_score = _count_zone_interactions(
        data, zone, atr
    )
    base = touch_count * 10 + rejection_count * 8 + displacement_score * 12
    base += zone.get("sweep_count", 0) * 15
    base += min(zone.get("cluster_point_count", 1), 6) * 3

    if level == "macro":
        raw = base + 35
        return int(min(100, max(70, raw)))
    if level == "mid":
        raw = base + 20
        return int(min(85, max(40, raw)))
    raw = base + 5
    return int(min(60, max(10, raw)))


def _macro_valid(zone: dict[str, Any]) -> bool:
    swing_touches = zone.get("swing_touch_count", zone.get("touch_count", 0))
    if swing_touches >= 2:
        return True
    if zone.get("sweep_count", 0) >= 1 and zone.get("rejection_count", 0) >= 1:
        return True
    return zone.get("strength", 0) >= 75


def _mid_valid(zone: dict[str, Any]) -> bool:
    return zone.get("rejection_count", 0) >= 1 or zone.get("has_reaction", False)


def _micro_valid(zone: dict[str, Any]) -> bool:
    return zone.get("has_wick_cluster", False) or zone.get("has_reversal", False)


def _overlaps_macro(zone: dict[str, Any], macro_zones: list[dict[str, Any]]) -> bool:
    for macro in macro_zones:
        if _zone_overlap_ratio(zone, macro) >= 0.3:
            return True
    return False


def _count_swing_touches(zone: dict[str, Any], swings: list[dict[str, Any]]) -> int:
    count = 0
    for swing in swings:
        price = float(swing["price"])
        if zone["low"] <= price <= zone["high"]:
            count += 1
    return count


def apply_institutional_layers(
    data: pd.DataFrame,
    grouped: dict[str, list[dict[str, Any]]],
    sweeps: list[dict[str, Any]],
    swings: list[dict[str, Any]],
    atr: pd.Series,
) -> dict[str, list[dict[str, Any]]]:
    """Apply layer-specific caps, strength bands, and validation rules."""
    config = DEFAULT_CONFIG
    enriched: dict[str, list[dict[str, Any]]] = {"macro": [], "mid": [], "micro": []}

    for level in ("macro", "mid", "micro"):
        for zone in grouped.get(level, []):
            item = dict(zone)
            item["mid"] = (item["low"] + item["high"]) / 2.0
            item["center"] = item["mid"]
            item["level"] = level
            item["level_rank"] = ZONE_LEVELS[level]
            touch_count, rejection_count, displacement_score = _count_zone_interactions(
                data, item, atr
            )
            item["touch_count"] = touch_count
            item["rejection_count"] = rejection_count
            item["displacement_score"] = displacement_score
            item["swing_touch_count"] = _count_swing_touches(item, swings)
            item["has_reaction"] = rejection_count >= 1
            item["has_wick_cluster"] = item.get("cluster_point_count", 0) >= 2 or level == "micro"
            item["has_reversal"] = displacement_score >= 1
            item["strength"] = _layer_strength(item, level, data, atr)
            if level == "macro":
                item["type"] = _macro_zone_type(item, data)
            enriched[level].append(item)

    macro = _merge_by_overlap(enriched["macro"], config.macro_merge_overlap)
    macro = [z for z in macro if _macro_valid(z)]
    macro.sort(key=lambda z: z["strength"], reverse=True)
    macro = macro[: config.macro_max_zones]

    mid_candidates = []
    for zone in enriched["mid"]:
        if not _mid_valid(zone):
            continue
        if _overlaps_macro(zone, macro) and zone.get("rejection_count", 0) < 1:
            continue
        mid_candidates.append(zone)

    mid = _merge_by_overlap(mid_candidates, 0.35)
    mid.sort(key=lambda z: z["strength"], reverse=True)
    mid = mid[: config.mid_max_zones]
    if len(mid) < config.mid_min_zones:
        pool = enriched["mid"] + [
            z for z in enriched["micro"] if z.get("strength", 0) >= 35
        ]
        mid = sorted(pool, key=lambda z: z["strength"], reverse=True)[: config.mid_min_zones]

    micro_candidates = [z for z in enriched["micro"] if _micro_valid(z)]
    if len(micro_candidates) < config.micro_min_zones:
        micro_candidates = enriched["micro"]
    micro = _merge_by_overlap(micro_candidates, 0.25)
    micro.sort(key=lambda z: z["strength"], reverse=True)
    micro = micro[: config.micro_max_zones]
    if len(micro) < config.micro_min_zones:
        micro = sorted(enriched["micro"], key=lambda z: z["strength"], reverse=True)[
            : config.micro_min_zones
        ]

    total = len(macro) + len(mid) + len(micro)
    if total < config.min_total_zones:
        pool = sorted(
            enriched["macro"] + enriched["mid"] + enriched["micro"],
            key=lambda z: z["strength"],
            reverse=True,
        )
        flat = flatten_zone_map({"macro": macro, "mid": mid, "micro": micro})
        existing_ids = {(z["level"], round(z["mid"], 5)) for z in flat}
        for zone in pool:
            key = (zone["level"], round(zone["mid"], 5))
            if key in existing_ids:
                continue
            enriched[zone["level"]].append(zone)
            existing_ids.add(key)
            total += 1
            if total >= config.min_total_zones:
                break
        macro = sorted(enriched["macro"], key=lambda z: z["strength"], reverse=True)[
            : config.macro_max_zones
        ]
        mid = sorted(enriched["mid"], key=lambda z: z["strength"], reverse=True)[
            : config.mid_max_zones
        ]
        micro = sorted(enriched["micro"], key=lambda z: z["strength"], reverse=True)[
            : config.micro_max_zones
        ]

    result = {"macro": macro, "mid": mid, "micro": micro}
    return _assign_nesting(result)


def format_zone_output(zone: dict[str, Any]) -> dict[str, Any]:
    """Public zone fields for strict output format."""
    return {
        "low": round(float(zone["low"]), 5),
        "high": round(float(zone["high"]), 5),
        "mid": round(float(zone.get("mid", zone["center"])), 5),
        "center": round(float(zone.get("center", zone["mid"])), 5),
        "strength": int(zone.get("strength", 0)),
        "type": zone.get("type", zone.get("level", "mid")),
        "level": zone.get("level"),
        "touch_count": int(zone.get("touch_count", 0)),
        "sweep_count": int(zone.get("sweep_count", 0)),
        "source": zone.get("source", "cluster"),
    }
