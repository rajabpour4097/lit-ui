"""ATR-based liquidity zone construction and clustering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG, EngineConfig
from liquidity_zone_engine.structure import detect_swings, has_break_of_structure
from liquidity_zone_engine.utils import normalize_ohlc


def _build_raw_zones(
    swings: list[dict[str, Any]],
    atr: pd.Series,
    zone_k: float,
) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []

    for swing in swings:
        idx = swing["index"]
        atr_value = float(atr.iloc[idx])
        if atr_value <= 0:
            continue

        swing_price = float(swing["price"])
        half_width = atr_value * zone_k
        leg_type = swing.get("leg_type", "corrective")

        zones.append(
            {
                "low": swing_price - half_width,
                "high": swing_price + half_width,
                "center": swing_price,
                "strength": 0,
                "type": leg_type,
                "touch_count": 0,
                "sweep_count": 0,
                "_atr": atr_value,
                "_index": idx,
                "_leg_type": leg_type,
                "_swing_type": swing["type"],
            }
        )

    return zones


def _zone_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap_low = max(left["low"], right["low"])
    overlap_high = min(left["high"], right["high"])
    if overlap_high <= overlap_low:
        return 0.0

    overlap = overlap_high - overlap_low
    smaller_height = min(left["high"] - left["low"], right["high"] - right["low"])
    if smaller_height <= 0:
        return 0.0
    return overlap / smaller_height


def _merge_two_zones(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    merged["low"] = min(current["low"], candidate["low"])
    merged["high"] = max(current["high"], candidate["high"])
    merged["center"] = (merged["low"] + merged["high"]) / 2.0
    merged["_atr"] = max(current["_atr"], candidate["_atr"])
    merged["_index"] = min(current["_index"], candidate["_index"])
    return merged


def _can_merge_zones(
    current: dict[str, Any],
    candidate: dict[str, Any],
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
    atr: pd.Series,
) -> bool:
    if current["_leg_type"] != candidate["_leg_type"]:
        return False

    if has_break_of_structure(
        data,
        current["_index"],
        candidate["_index"],
        swings,
        atr=atr,
    ):
        return False

    overlap = _zone_overlap_ratio(current, candidate)
    if overlap > DEFAULT_CONFIG.zone_overlap_merge_pct:
        return True

    reference_atr = max(current["_atr"], candidate["_atr"])
    distance = abs(candidate["center"] - current["center"])
    return distance <= reference_atr * DEFAULT_CONFIG.zone_merge_atr


def _compress_zone_group(
    zones: list[dict[str, Any]],
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
    atr: pd.Series,
) -> list[dict[str, Any]]:
    if not zones:
        return []

    result = [dict(zone) for zone in zones]
    merged_any = True

    while merged_any:
        merged_any = False
        next_result: list[dict[str, Any]] = []
        consumed = [False] * len(result)

        for i in range(len(result)):
            if consumed[i]:
                continue

            current = dict(result[i])
            for j in range(i + 1, len(result)):
                if consumed[j]:
                    continue
                candidate = result[j]
                if _can_merge_zones(current, candidate, data, swings, atr):
                    current = _merge_two_zones(current, candidate)
                    consumed[j] = True
                    merged_any = True

            next_result.append(current)

        result = next_result

    result.sort(key=lambda zone: zone["_index"])
    return result


def _cluster_zones(
    zones: list[dict[str, Any]],
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
    atr: pd.Series,
) -> list[dict[str, Any]]:
    if not zones:
        return []

    merged: list[dict[str, Any]] = []
    for leg_type in ("impulsive", "corrective"):
        for swing_type in ("swing_high", "swing_low"):
            group = [
                zone
                for zone in zones
                if zone["_leg_type"] == leg_type and zone["_swing_type"] == swing_type
            ]
            if not group:
                continue
            merged.extend(
                _compress_zone_group(
                    sorted(group, key=lambda zone: zone["_index"]),
                    data,
                    swings,
                    atr,
                )
            )

    merged.sort(key=lambda zone: zone["_index"])
    return merged


def _split_oversized_zones(
    zones: list[dict[str, Any]],
    swings: list[dict[str, Any]],
    atr: pd.Series,
    zone_k: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for zone in zones:
        height = zone["high"] - zone["low"]
        max_height = zone["_atr"] * DEFAULT_CONFIG.max_zone_height_atr

        if height <= max_height:
            result.append(zone)
            continue

        inner_swings = [
            s
            for s in swings
            if zone["low"] <= s["price"] <= zone["high"]
            and s.get("leg_type", "corrective") == zone["_leg_type"]
        ]

        if len(inner_swings) < 2:
            half_height = height / 4.0
            for sub_center in (zone["low"] + half_height, zone["high"] - half_height):
                sub = dict(zone)
                sub["low"] = sub_center - half_height / 2.0
                sub["high"] = sub_center + half_height / 2.0
                sub["center"] = sub_center
                result.append(sub)
            continue

        result.extend(_build_raw_zones(inner_swings, atr, zone_k=zone_k))

    return result


def _count_touch_events(data: pd.DataFrame, zone: dict[str, Any]) -> int:
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()

    touches = 0
    in_zone = False

    for i in range(len(data)):
        bar_intersects = lows[i] <= zone["high"] and highs[i] >= zone["low"]
        if bar_intersects and not in_zone:
            touches += 1
            in_zone = True
        elif not bar_intersects:
            in_zone = False

    return touches


def _count_zone_interactions(
    data: pd.DataFrame,
    zone: dict[str, Any],
    atr: pd.Series,
) -> tuple[int, int, int]:
    """Return touch_count, rejection_count, displacement_score."""
    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()
    zone_low = zone["low"]
    zone_high = zone["high"]

    touch_count = _count_touch_events(data, zone)
    rejection_count = 0
    displacement_score = 0

    for i in range(len(data)):
        if highs[i] < zone_low or lows[i] > zone_high:
            continue

        atr_value = float(atr.iloc[i])
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        body = abs(closes[i] - opens[i]) or 1e-9

        buy_side_probe = highs[i] > zone_high and zone_low <= closes[i] <= zone_high
        sell_side_probe = lows[i] < zone_low and zone_low <= closes[i] <= zone_high

        if buy_side_probe and upper_wick >= body * 0.5:
            rejection_count += 1
            if i + 1 < len(data) and closes[i + 1] < closes[i] - atr_value * 0.25:
                displacement_score += 1

        if sell_side_probe and lower_wick >= body * 0.5:
            rejection_count += 1
            if i + 1 < len(data) and closes[i + 1] > closes[i] + atr_value * 0.25:
                displacement_score += 1

    return touch_count, rejection_count, displacement_score


def _time_decay_score(data: pd.DataFrame, zone: dict[str, Any]) -> float:
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    last_touch = -1

    for i in range(len(data)):
        if lows[i] <= zone["high"] and highs[i] >= zone["low"]:
            last_touch = i

    if last_touch < 0:
        return 0.0

    bars_since = (len(data) - 1) - last_touch
    return max(0.0, 12.0 - float(bars_since))


def _normalize_strength(raw_scores: list[float]) -> list[int]:
    if not raw_scores:
        return []

    cap = max(raw_scores) * 0.85 if max(raw_scores) > 0 else 1.0
    capped = [min(score, cap) for score in raw_scores]
    max_score = max(capped)

    if max_score <= 0:
        return [20 for _ in capped]

    normalized = [int(round((score / max_score) * 75 + 20)) for score in capped]
    return [min(95, max(15, value)) for value in normalized]


def finalize_zones(
    data: pd.DataFrame,
    zones: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    atr: pd.Series,
) -> list[dict[str, Any]]:
    """Compute touch/sweep counts and normalized strength for output zones."""
    config = DEFAULT_CONFIG
    sweep_counts: dict[int, int] = {}
    for sweep in sweeps:
        zone_index = sweep["zone_index"]
        sweep_counts[zone_index] = sweep_counts.get(zone_index, 0) + 1

    enriched: list[dict[str, Any]] = []

    for zone_index, zone in enumerate(zones):
        touch_count, rejection_count, displacement_score = _count_zone_interactions(
            data, zone, atr
        )
        time_score = _time_decay_score(data, zone)
        sweep_count = sweep_counts.get(zone_index, 0)

        raw_score = (
            rejection_count * config.strength_weight_rejection
            + sweep_count * config.strength_weight_sweep
            + displacement_score * config.strength_weight_displacement
            + time_score * config.strength_weight_time
        )

        enriched.append(
            {
                **zone,
                "touch_count": touch_count,
                "sweep_count": sweep_count,
                "_rejection_count": rejection_count,
                "_displacement_score": displacement_score,
                "_time_score": time_score,
                "_raw_strength": float(raw_score),
            }
        )

    strengths = _normalize_strength([zone["_raw_strength"] for zone in enriched])

    output: list[dict[str, Any]] = []
    for zone, strength in zip(enriched, strengths):
        output.append(
            {
                "low": float(zone["low"]),
                "high": float(zone["high"]),
                "center": float(zone["center"]),
                "strength": strength,
                "type": zone["type"],
                "touch_count": int(zone["touch_count"]),
                "sweep_count": int(zone["sweep_count"]),
                "_displacement_score": int(zone["_displacement_score"]),
            }
        )

    return output


def _passes_quality_gate(zone: dict[str, Any]) -> bool:
    if zone["touch_count"] >= 2:
        return True
    if zone.get("_displacement_score", 0) > 0:
        return True
    if zone["sweep_count"] > 0:
        return True
    return False


def filter_quality_zones(
    zones: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove noise zones and remap sweep indices to the compressed set."""
    config = DEFAULT_CONFIG
    kept: list[dict[str, Any]] = []
    index_map: dict[int, int] = {}

    for old_index, zone in enumerate(zones):
        retested = zone["touch_count"] >= 2
        if zone["strength"] < config.min_zone_strength and not retested:
            continue
        if not _passes_quality_gate(zone):
            continue

        cleaned = {
            key: value for key, value in zone.items() if not key.startswith("_")
        }
        index_map[old_index] = len(kept)
        kept.append(cleaned)

    remapped_sweeps: list[dict[str, Any]] = []
    for sweep in sweeps:
        old_index = sweep["zone_index"]
        if old_index not in index_map:
            continue
        remapped_sweeps.append({**sweep, "zone_index": index_map[old_index]})

    return kept, remapped_sweeps


def _remove_redundant_overlaps(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not zones:
        return []

    ranked = sorted(zones, key=lambda zone: (zone["strength"], zone["touch_count"]), reverse=True)
    kept: list[dict[str, Any]] = []

    for candidate in ranked:
        redundant = False
        for existing in kept:
            if (
                existing["type"] == candidate["type"]
                and _zone_overlap_ratio(existing, candidate) > DEFAULT_CONFIG.zone_overlap_merge_pct
                and existing["strength"] >= candidate["strength"]
            ):
                redundant = True
                break
        if not redundant:
            kept.append(candidate)

    kept.sort(key=lambda zone: zone["center"])
    return kept


def tag_sub_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark nested zones as sub-zones and collapse them from primary output."""
    tagged = [dict(zone) for zone in zones]

    for i, child in enumerate(tagged):
        child_area = child["high"] - child["low"]
        for j, parent in enumerate(tagged):
            if i == j:
                continue

            parent_area = parent["high"] - parent["low"]
            contained = (
                child["low"] >= parent["low"]
                and child["high"] <= parent["high"]
                and parent_area > child_area
            )
            if contained and parent["strength"] >= child["strength"]:
                child["role"] = "sub-zone"
                break

    return [zone for zone in tagged if zone.get("role") != "sub-zone"]


def _prepare_zone_candidates(
    data: pd.DataFrame,
    atr: pd.Series,
    swing_points: list[dict[str, Any]],
    zone_k: float,
    cluster_atr_mult: float,
) -> list[dict[str, Any]]:
    del cluster_atr_mult  # compression uses config.zone_merge_atr

    raw_zones = _build_raw_zones(swing_points, atr, zone_k=zone_k)
    clustered = _cluster_zones(raw_zones, data, swing_points, atr)
    split = _split_oversized_zones(clustered, swing_points, atr, zone_k=zone_k)

    if len(split) != len(clustered):
        split = _cluster_zones(split, data, swing_points, atr)

    return split


def build_zones(
    df: pd.DataFrame,
    swing_window: int = DEFAULT_CONFIG.swing_window,
    atr_period: int = DEFAULT_CONFIG.atr_period,
    zone_k: float = DEFAULT_CONFIG.zone_k,
    cluster_atr_mult: float = DEFAULT_CONFIG.cluster_atr_mult,
    swings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build clustered liquidity zones around swing points.

    Zone width: swing_price +/- (ATR * zone_k)
    Structure-aware compression preserves separate impulsive/corrective legs.
    """
    EngineConfig(
        swing_window=swing_window,
        atr_period=atr_period,
        zone_k=zone_k,
        cluster_atr_mult=cluster_atr_mult,
    )

    data = normalize_ohlc(df)
    atr = compute_atr(data, period=atr_period)
    swing_points = swings if swings is not None else detect_swings(data, swing_window=swing_window)

    candidates = _prepare_zone_candidates(
        data, atr, swing_points, zone_k=zone_k, cluster_atr_mult=cluster_atr_mult
    )
    finalized = finalize_zones(data, candidates, sweeps=[], atr=atr)
    filtered, _ = filter_quality_zones(finalized, [])
    deduped = _remove_redundant_overlaps(filtered)
    return tag_sub_zones(deduped)
