"""Fractal multi-layer liquidity zone construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG, ZONE_LEVELS, EngineConfig
from liquidity_zone_engine.structure import detect_swings, has_break_of_structure
from liquidity_zone_engine.sweeps import detect_sweeps
from liquidity_zone_engine.utils import normalize_ohlc

LEVEL_ZONE_K = {"macro": 0.75, "mid": 0.6, "micro": 0.45}


def flatten_zone_map(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten grouped zones to a list preserving level metadata."""
    flat: list[dict[str, Any]] = []
    for level in ("macro", "mid", "micro"):
        for layer_index, zone in enumerate(grouped.get(level, [])):
            flat.append({**zone, "level": level, "layer_index": layer_index})
    return flat


def regroup_zones(flat: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Regroup flat zone list by level."""
    grouped: dict[str, list[dict[str, Any]]] = {"macro": [], "mid": [], "micro": []}
    for zone in flat:
        level = zone.get("level", "mid")
        if level not in grouped:
            grouped[level] = []
        cleaned = {k: v for k, v in zone.items() if not k.startswith("_")}
        grouped[level].append(cleaned)
    return grouped


def _zone_from_point(
    price: float,
    index: int,
    atr_value: float,
    level: str,
    leg_type: str,
    swing_type: str,
) -> dict[str, Any]:
    half = atr_value * LEVEL_ZONE_K.get(level, DEFAULT_CONFIG.zone_k)
    return {
        "low": price - half,
        "high": price + half,
        "center": price,
        "strength": 0,
        "type": leg_type,
        "level": level,
        "level_rank": ZONE_LEVELS[level],
        "touch_count": 0,
        "sweep_count": 0,
        "_atr": atr_value,
        "_index": index,
        "_leg_type": leg_type,
        "_swing_type": swing_type,
    }


def _swing_displacement(
    swing: dict[str, Any],
    prev_opposite: dict[str, Any] | None,
) -> float:
    if prev_opposite is None:
        return float("inf")
    return abs(float(swing["price"]) - float(prev_opposite["price"]))


def _is_idm_swing(swing: dict[str, Any]) -> bool:
    return bool(swing.get("is_idm")) or swing.get("structure_role") == "idm"


def _liquidity_point(
    price: float,
    index: int,
    atr_value: float,
    swing_type: str,
    leg_type: str,
    source: str,
) -> dict[str, Any]:
    return {
        "price": price,
        "index": index,
        "_atr": atr_value,
        "_swing_type": swing_type,
        "_leg_type": leg_type,
        "_source": source,
    }


def _collect_swing_points(
    swings: list[dict[str, Any]],
    atr: pd.Series,
    min_displacement_atr: float,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    last_high: dict[str, Any] | None = None
    last_low: dict[str, Any] | None = None

    for swing in swings:
        if _is_idm_swing(swing):
            continue

        idx = int(swing["index"])
        atr_value = float(atr.iloc[idx])
        prev = last_low if swing["type"] == "swing_high" else last_high
        displacement = _swing_displacement(swing, prev)

        if displacement < atr_value * min_displacement_atr:
            if swing["type"] == "swing_high":
                last_high = swing
            else:
                last_low = swing
            continue

        points.append(
            _liquidity_point(
                float(swing["price"]),
                idx,
                atr_value,
                swing["type"],
                swing.get("leg_type", "corrective"),
                "swing",
            )
        )

        if swing["type"] == "swing_high":
            last_high = swing
        else:
            last_low = swing

    return points


def _collect_wick_points(data: pd.DataFrame, atr: pd.Series) -> list[dict[str, Any]]:
    config = DEFAULT_CONFIG
    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()
    points: list[dict[str, Any]] = []

    for i in range(len(data)):
        body = abs(closes[i] - opens[i]) or 1e-9
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        atr_value = float(atr.iloc[i])

        if upper_wick >= body * config.sweep_wick_body_ratio:
            points.append(
                _liquidity_point(
                    float(highs[i]),
                    i,
                    atr_value,
                    "swing_high",
                    "corrective",
                    "wick",
                )
            )

        if lower_wick >= body * config.sweep_wick_body_ratio:
            points.append(
                _liquidity_point(
                    float(lows[i]),
                    i,
                    atr_value,
                    "swing_low",
                    "corrective",
                    "wick",
                )
            )

    return points


def _collect_equal_level_points(data: pd.DataFrame, atr: pd.Series) -> list[dict[str, Any]]:
    config = DEFAULT_CONFIG
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    points: list[dict[str, Any]] = []

    for i in range(len(data)):
        atr_value = float(atr.iloc[i])
        tol = atr_value * config.micro_equal_level_atr

        for j in range(max(0, i - 12), min(len(data), i + 12)):
            if j == i:
                continue
            if abs(highs[j] - highs[i]) <= tol:
                points.append(
                    _liquidity_point(
                        float(highs[i]),
                        i,
                        atr_value,
                        "swing_high",
                        "corrective",
                        "equal_high",
                    )
                )
                break

        for j in range(max(0, i - 12), min(len(data), i + 12)):
            if j == i:
                continue
            if abs(lows[j] - lows[i]) <= tol:
                points.append(
                    _liquidity_point(
                        float(lows[i]),
                        i,
                        atr_value,
                        "swing_low",
                        "corrective",
                        "equal_low",
                    )
                )
                break

    return points


def _cluster_points(
    points: list[dict[str, Any]],
    cluster_mult: float,
    min_points: int | None = None,
) -> list[list[dict[str, Any]]]:
    if not points:
        return []

    config = DEFAULT_CONFIG
    required = config.cluster_min_points if min_points is None else min_points
    ordered = sorted(points, key=lambda item: (item["price"], item["index"]))
    clusters: list[list[dict[str, Any]]] = []
    current = [ordered[0]]

    for point in ordered[1:]:
        ref_atr = max(member["_atr"] for member in current)
        threshold = ref_atr * cluster_mult
        if abs(point["price"] - current[-1]["price"]) <= threshold:
            current.append(point)
            continue

        if len(current) >= required:
            clusters.append(current)
        current = [point]

    if len(current) >= required:
        clusters.append(current)

    return clusters


def _dominant(values: list[str]) -> str:
    return max(set(values), key=values.count)


def _zone_from_cluster(cluster: list[dict[str, Any]], level: str) -> dict[str, Any]:
    prices = [point["price"] for point in cluster]
    indices = [point["index"] for point in cluster]
    atr_value = sum(point["_atr"] for point in cluster) / len(cluster)
    center = sum(prices) / len(prices)
    idx = min(indices)
    half = max(atr_value * LEVEL_ZONE_K.get(level, DEFAULT_CONFIG.zone_k), (max(prices) - min(prices)) / 2.0)

    return {
        "low": center - half,
        "high": center + half,
        "center": center,
        "strength": 0,
        "type": _dominant([point["_leg_type"] for point in cluster]),
        "level": level,
        "level_rank": ZONE_LEVELS[level],
        "touch_count": 0,
        "sweep_count": 0,
        "source": "cluster",
        "cluster_point_count": len(cluster),
        "_atr": atr_value,
        "_index": idx,
        "_leg_type": _dominant([point["_leg_type"] for point in cluster]),
        "_swing_type": _dominant([point["_swing_type"] for point in cluster]),
    }


def _collect_range_boundaries(data: pd.DataFrame, atr: pd.Series) -> list[dict[str, Any]]:
    """Major range boundaries as macro liquidity anchors."""
    config = DEFAULT_CONFIG
    window = min(80, len(data))
    if window < 10:
        return []

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    start = len(data) - window
    idx = len(data) - 1
    atr_value = float(atr.iloc[idx])
    range_high = float(highs[start:].max())
    range_low = float(lows[start:].min())

    return [
        _liquidity_point(range_high, idx, atr_value, "swing_high", "impulsive", "range"),
        _liquidity_point(range_low, idx, atr_value, "swing_low", "impulsive", "range"),
        _liquidity_point((range_high + range_low) / 2.0, idx, atr_value, "swing_low", "corrective", "range"),
    ]


def _collect_reaction_points(data: pd.DataFrame, atr: pd.Series) -> list[dict[str, Any]]:
    config = DEFAULT_CONFIG
    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()
    points: list[dict[str, Any]] = []

    for i in range(1, len(data)):
        body = abs(closes[i] - opens[i]) or 1e-9
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        atr_value = float(atr.iloc[i])

        if upper_wick >= body * 0.8 or lower_wick >= body * 0.8:
            price = float(highs[i] if upper_wick >= lower_wick else lows[i])
            points.append(
                _liquidity_point(
                    price,
                    i,
                    atr_value,
                    "swing_high" if upper_wick >= lower_wick else "swing_low",
                    "corrective",
                    "reaction",
                )
            )

    return points


def _generate_macro_zones(
    swings: list[dict[str, Any]],
    atr: pd.Series,
    data: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    config = DEFAULT_CONFIG
    points = _collect_swing_points(swings, atr, config.macro_swing_atr)
    if data is not None:
        points.extend(_collect_range_boundaries(data, atr))
    clusters = _cluster_points(points, config.cluster_atr_mult * 2.0, min_points=1)
    return [_zone_from_cluster(cluster, "macro") for cluster in clusters]


def _generate_mid_zones(
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
    atr: pd.Series,
    macro_zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    config = DEFAULT_CONFIG
    points = _collect_swing_points(swings, atr, config.mid_swing_atr * 0.5)
    points.extend(_collect_reaction_points(data, atr))

    if macro_zones:
        filtered: list[dict[str, Any]] = []
        for point in points:
            inside_macro = any(
                macro["low"] <= point["price"] <= macro["high"] for macro in macro_zones
            )
            if inside_macro:
                filtered.append(point)
        points = filtered

    clusters = _cluster_points(points, config.cluster_atr_mult * 1.5, min_points=1)
    return [_zone_from_cluster(cluster, "mid") for cluster in clusters]


def _generate_micro_zones(data: pd.DataFrame, atr: pd.Series) -> list[dict[str, Any]]:
    config = DEFAULT_CONFIG
    points = _collect_wick_points(data, atr) + _collect_equal_level_points(data, atr)
    clusters = _cluster_points(
        points,
        config.cluster_atr_mult,
        min_points=config.micro_cluster_min_points,
    )
    return [_zone_from_cluster(cluster, "micro") for cluster in clusters]


def _can_merge_same_level(
    left: dict[str, Any],
    right: dict[str, Any],
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
    atr: pd.Series,
) -> bool:
    if left.get("level") != right.get("level"):
        return False
    if left["_leg_type"] != right["_leg_type"]:
        return False
    if left["_swing_type"] != right["_swing_type"]:
        return False
    if has_break_of_structure(data, left["_index"], right["_index"], swings, atr=atr):
        return False

    reference_atr = max(left["_atr"], right["_atr"])
    distance = abs(left["center"] - right["center"])
    return distance <= reference_atr * DEFAULT_CONFIG.same_level_merge_atr


def _merge_within_level(
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
                if _can_merge_same_level(current, candidate, data, swings, atr):
                    current["low"] = min(current["low"], candidate["low"])
                    current["high"] = max(current["high"], candidate["high"])
                    current["center"] = (current["low"] + current["high"]) / 2.0
                    current["_atr"] = max(current["_atr"], candidate["_atr"])
                    current["_index"] = min(current["_index"], candidate["_index"])
                    current["source"] = "cluster"
                    current["cluster_point_count"] = int(
                        current.get("cluster_point_count", 1)
                        + candidate.get("cluster_point_count", 1)
                    )
                    consumed[j] = True
                    merged_any = True
            next_result.append(current)

        result = next_result

    result.sort(key=lambda zone: zone["_index"])
    return result


def _assign_nesting(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Link mid zones to macro parents and micro zones to mid/macro parents."""
    for level, zones in grouped.items():
        for zone in zones:
            zone["parent_macro"] = None
            zone["parent_mid"] = None

    for mid in grouped["mid"]:
        for macro in grouped["macro"]:
            if mid["low"] >= macro["low"] and mid["high"] <= macro["high"]:
                mid["parent_macro"] = macro["center"]
                break

    for micro in grouped["micro"]:
        for mid in grouped["mid"]:
            if micro["low"] >= mid["low"] and micro["high"] <= mid["high"]:
                micro["parent_mid"] = mid["center"]
                break
        if micro["parent_mid"] is None:
            for macro in grouped["macro"]:
                if micro["low"] >= macro["low"] and micro["high"] <= macro["high"]:
                    micro["parent_macro"] = macro["center"]
                    break

    return grouped


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


def _last_touch_bar(data: pd.DataFrame, zone: dict[str, Any]) -> int:
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    last_touch = -1

    for i in range(len(data)):
        if lows[i] <= zone["high"] and highs[i] >= zone["low"]:
            last_touch = i

    return last_touch


def _last_activity_bar(
    data: pd.DataFrame,
    zone_index: int,
    sweeps: list[dict[str, Any]],
    zone: dict[str, Any],
) -> int:
    last_touch = _last_touch_bar(data, zone)
    last_sweep = -1
    for sweep in sweeps:
        if sweep.get("zone_index") == zone_index:
            last_sweep = max(last_sweep, int(sweep.get("bar_index", -1)))
    return max(last_touch, last_sweep)


def filter_tradable_zones(
    data: pd.DataFrame,
    zones: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Keep only zones with proven liquidity activity:
    - minimum touches on price
    - at least one real sweep
    - recent activity within max_inactive_bars
    """
    config = DEFAULT_CONFIG
    kept: list[dict[str, Any]] = []
    index_map: dict[int, int] = {}

    for zone_index, zone in enumerate(zones):
        if zone.get("touch_count", 0) < config.min_zone_touches:
            continue
        if zone.get("sweep_count", 0) < config.min_zone_sweeps:
            continue

        last_activity = _last_activity_bar(data, zone_index, sweeps, zone)
        if last_activity < 0:
            continue

        bars_since = (len(data) - 1) - last_activity
        if bars_since > config.max_inactive_bars:
            continue

        index_map[zone_index] = len(kept)
        kept.append(zone)

    remapped_sweeps: list[dict[str, Any]] = []
    for sweep in sweeps:
        old_index = sweep.get("zone_index")
        if old_index not in index_map:
            continue
        item = dict(sweep)
        item["zone_index"] = index_map[old_index]
        remapped_sweeps.append(item)

    return kept, remapped_sweeps


def finalize_zones(
    data: pd.DataFrame,
    zones: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    atr: pd.Series,
) -> list[dict[str, Any]]:
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
                "level": zone.get("level", "mid"),
                "level_rank": zone.get("level_rank", ZONE_LEVELS.get(zone.get("level", "mid"), 2)),
                "touch_count": int(zone["touch_count"]),
                "sweep_count": int(zone["sweep_count"]),
                "source": zone.get("source", "cluster"),
                "cluster_point_count": int(zone.get("cluster_point_count", 0)),
                "parent_macro": zone.get("parent_macro"),
                "parent_mid": zone.get("parent_mid"),
            }
        )

    return output


def build_fractal_zone_map(
    data: pd.DataFrame,
    swings: list[dict[str, Any]],
    atr: pd.Series,
) -> dict[str, list[dict[str, Any]]]:
    """Build nested macro / mid / micro zone layers without cross-level merging."""
    macro = _merge_within_level(_generate_macro_zones(swings, atr, data), data, swings, atr)
    mid = _merge_within_level(
        _generate_mid_zones(data, swings, atr, macro), data, swings, atr
    )
    micro = _merge_within_level(_generate_micro_zones(data, atr), data, swings, atr)

    grouped = {"macro": macro, "mid": mid, "micro": micro}
    return _assign_nesting(grouped)


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


def has_nested_overlap(grouped: dict[str, list[dict[str, Any]]]) -> bool:
    """Return True when mid/micro layers nest inside macro/mid zones."""
    if not grouped["macro"] or not grouped["mid"]:
        return False

    mid_in_macro = any(
        any(
            mid["low"] >= macro["low"] and mid["high"] <= macro["high"]
            for macro in grouped["macro"]
        )
        for mid in grouped["mid"]
    )
    micro_nested = any(
        zone.get("parent_mid") is not None or zone.get("parent_macro") is not None
        for zone in grouped["micro"]
    )
    return mid_in_macro or micro_nested


def build_zones(
    df: pd.DataFrame,
    swing_window: int = DEFAULT_CONFIG.swing_window,
    atr_period: int = DEFAULT_CONFIG.atr_period,
    zone_k: float = DEFAULT_CONFIG.zone_k,
    cluster_atr_mult: float = DEFAULT_CONFIG.cluster_atr_mult,
    swings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build fractal zones and return flattened list (backward-compatible API)."""
    EngineConfig(
        swing_window=swing_window,
        atr_period=atr_period,
        zone_k=zone_k,
        cluster_atr_mult=cluster_atr_mult,
    )

    data = normalize_ohlc(df)
    atr = compute_atr(data, period=atr_period)
    swing_points = swings if swings is not None else detect_swings(data, swing_window=swing_window)

    grouped = build_fractal_zone_map(data, swing_points, atr)
    flat = flatten_zone_map(grouped)
    prelim = finalize_zones(data, flat, sweeps=[], atr=atr)
    sweeps = detect_sweeps(df, prelim)
    finalized = finalize_zones(data, prelim, sweeps, atr=atr)
    tradable, _ = filter_tradable_zones(data, finalized, sweeps)
    return tradable
