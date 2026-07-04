"""LIT (Liquidity Inducement Theory) — institutional structure reconstruction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc
from liquidity_zone_engine.zones import _count_zone_interactions, _zone_overlap_ratio


def _find_equal_levels(swings: list[dict[str, Any]], tolerance_pct: float = 0.00015) -> list[dict[str, Any]]:
    pools: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {"swing_high": [], "swing_low": []}
    for swing in swings:
        grouped[swing["type"]].append(swing)

    for swing_type, items in grouped.items():
        for i, a in enumerate(items):
            cluster = [a]
            for b in items[i + 1 :]:
                ref = max(abs(a["price"]), 1e-9)
                if abs(a["price"] - b["price"]) / ref <= tolerance_pct:
                    cluster.append(b)
            if len(cluster) >= 2:
                price = sum(float(s["price"]) for s in cluster) / len(cluster)
                pools.append(
                    {
                        "price": price,
                        "type": swing_type,
                        "source": "equal_level",
                        "pool_type": "EQH" if swing_type == "swing_high" else "EQL",
                    }
                )
    return pools


def _major_swing_extremes(swings: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
    highs = sorted(
        [s for s in swings if s["type"] == "swing_high"],
        key=lambda s: float(s["price"]),
        reverse=True,
    )[:top_n]
    lows = sorted(
        [s for s in swings if s["type"] == "swing_low"],
        key=lambda s: float(s["price"]),
    )[:top_n]
    pools: list[dict[str, Any]] = []
    for swing in highs + lows:
        pools.append(
            {
                "price": float(swing["price"]),
                "type": swing["type"],
                "source": "major_swing",
                "pool_type": "swing_high" if swing["type"] == "swing_high" else "swing_low",
            }
        )
    return pools


def classify_lit_regime(bos_timeline: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Trend ONLY from last 2 valid BOS same direction with no opposite BOS after.
    Default = range / neutral bias.
    """
    bos_events = [e for e in bos_timeline if e.get("type") == "BOS" and e.get("validity") == "confirmed"]

    if len(bos_events) < 2:
        return {"regime": "range", "bias": "neutral"}

    last_two = bos_events[-2:]
    if last_two[0]["direction"] != last_two[1]["direction"]:
        return {"regime": "range", "bias": "neutral"}

    direction = last_two[0]["direction"]
    last_bos_index = int(last_two[-1].get("index", 0))

    if any(
        e["direction"] != direction and int(e.get("index", 0)) > last_bos_index
        for e in bos_events
    ):
        return {"regime": "range", "bias": "neutral"}

    if direction == "bullish":
        return {"regime": "trending_bull", "bias": "bullish"}
    return {"regime": "trending_bear", "bias": "bearish"}


def calibrate_bos_choch(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate levels from BOS timeline."""
    cleaned: list[dict[str, Any]] = []
    seen: list[float] = []
    for event in timeline:
        level = float(event["level"])
        if any(abs(level - s) <= level * 0.0001 + 0.00005 for s in seen):
            continue
        cleaned.append(event)
        seen.append(level)
    return cleaned


def _count_rejection_interactions(
    data: pd.DataFrame,
    zone: dict[str, Any],
    atr: pd.Series,
) -> tuple[int, int]:
    """Count only rejection-after-touch interactions (not raw touches)."""
    _, rejection_count, displacement_score = _count_zone_interactions(data, zone, atr)
    return rejection_count, displacement_score


def _recency_score(data: pd.DataFrame, zone: dict[str, Any], window: int = 20) -> float:
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    last_touch = -1
    for i in range(len(data)):
        if lows[i] <= zone["high"] and highs[i] >= zone["low"]:
            last_touch = i
    if last_touch < 0:
        return 0.0
    bars_since = (len(data) - 1) - last_touch
    return max(0.0, 1.0 - bars_since / window)


def _bos_confluence(zone: dict[str, Any], bos_timeline: list[dict[str, Any]]) -> float:
    if not bos_timeline:
        return 0.0
    last_bos = next((e for e in reversed(bos_timeline) if e.get("type") == "BOS"), None)
    if not last_bos:
        return 0.0
    level = float(last_bos["level"])
    if zone["low"] <= level <= zone["high"]:
        return 1.0
    return 0.0


def _compute_zone_strength(
    zone: dict[str, Any],
    data: pd.DataFrame,
    atr: pd.Series,
    bos_timeline: list[dict[str, Any]],
) -> float:
    """40% sweeps, 30% displacement, 20% recency, 10% BOS confluence."""
    sweep_part = min(int(zone.get("real_sweep_count", 0)), 4) / 4.0 * 40.0
    disp_part = min(int(zone.get("displacement_score", 0)), 4) / 4.0 * 30.0
    rec_part = _recency_score(data, zone) * 20.0
    bos_part = _bos_confluence(zone, bos_timeline) * 10.0
    return sweep_part + disp_part + rec_part + bos_part


def _normalize_zone_strengths(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not zones:
        return []

    scored = sorted(zones, key=lambda z: z.get("_raw_strength", 0), reverse=True)
    top_idx = 0
    output: list[dict[str, Any]] = []

    for i, zone in enumerate(scored):
        item = dict(zone)
        raw = float(item.get("_raw_strength", 0))
        if i == top_idx and raw > 0:
            item["strength"] = int(min(95, max(81, round(raw))))
        else:
            item["strength"] = int(min(70, max(20, round(raw * 0.7 + 20))))
        output.append(item)

    output.sort(key=lambda z: z["center"], reverse=True)
    return output


def _merge_zones(zones: list[dict[str, Any]], min_overlap: float) -> list[dict[str, Any]]:
    ordered = sorted(zones, key=lambda z: z.get("_raw_strength", 0), reverse=True)
    merged: list[dict[str, Any]] = []
    for zone in ordered:
        absorbed = False
        for existing in merged:
            if _zone_overlap_ratio(existing, zone) >= min_overlap:
                existing["low"] = min(existing["low"], zone["low"])
                existing["high"] = max(existing["high"], zone["high"])
                existing["center"] = (existing["low"] + existing["high"]) / 2.0
                existing["mid"] = existing["center"]
                existing["real_sweep_count"] = int(existing.get("real_sweep_count", 0)) + int(
                    zone.get("real_sweep_count", 0)
                )
                existing["displacement_score"] = int(existing.get("displacement_score", 0)) + int(
                    zone.get("displacement_score", 0)
                )
                existing["valid_interactions"] = int(existing.get("valid_interactions", 0)) + int(
                    zone.get("valid_interactions", 0)
                )
                absorbed = True
                break
        if not absorbed:
            merged.append(dict(zone))
    return merged


def build_lit_zones(
    df: pd.DataFrame,
    grouped: dict[str, list[dict[str, Any]]],
    swings: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    atr: pd.Series,
    bos_timeline: list[dict[str, Any]],
    timeframe: str | None,
) -> list[dict[str, Any]]:
    config = DEFAULT_CONFIG
    data = normalize_ohlc(df)
    max_zones = config.lit_max_zones_h1 if timeframe == "H1" else config.lit_max_zones_default
    atr_last = float(atr.iloc[-1])
    tol = atr_last * config.swing_alignment_atr

    candidates: list[dict[str, Any]] = []
    for zone in grouped.get("macro", []):
        item = dict(zone)
        item["level"] = "macro"
        item["type"] = "macro"
        item["source"] = "macro_swing"
        item["mid"] = (item["low"] + item["high"]) / 2.0
        item["center"] = item["mid"]
        candidates.append(item)

    for pool in _find_equal_levels(swings):
        half = atr_last * config.zone_k * 0.45
        price = float(pool["price"])
        candidates.append(
            {
                "low": price - half,
                "high": price + half,
                "mid": price,
                "center": price,
                "level": "mid",
                "type": "mid",
                "source": pool["source"],
                "pool_type": pool["pool_type"],
            }
        )

    filtered: list[dict[str, Any]] = []
    for zone in candidates:
        rejections, displacement = _count_rejection_interactions(data, zone, atr)
        zone["valid_interactions"] = rejections
        zone["displacement_score"] = displacement
        zone["real_sweep_count"] = sum(
            1
            for s in sweeps
            if abs(float(s.get("liquidity_price", 0)) - zone["center"]) <= tol
        )
        if zone["valid_interactions"] < config.lit_min_zone_interactions:
            continue
        if zone["real_sweep_count"] < 1:
            continue
        zone["_raw_strength"] = _compute_zone_strength(zone, data, atr, bos_timeline)
        filtered.append(zone)

    merged = _merge_zones(filtered, config.lit_zone_merge_overlap)
    merged.sort(key=lambda z: z.get("_raw_strength", 0), reverse=True)
    return _normalize_zone_strengths(merged[:max_zones])


def liquidity_pools_from_structure(
    swings: list[dict[str, Any]],
    last_close: float,
) -> dict[str, Any]:
    """Structural pools: EQH above price, EQL below price."""
    eqh = _find_equal_levels(swings)
    majors = _major_swing_extremes(swings)

    buy_pools = [
        p for p in eqh + majors
        if p["type"] == "swing_high" and float(p["price"]) > last_close
    ]
    sell_pools = [
        p for p in eqh + majors
        if p["type"] == "swing_low" and float(p["price"]) < last_close
    ]

    buy_pools.sort(key=lambda p: float(p["price"]))
    sell_pools.sort(key=lambda p: float(p["price"]), reverse=True)

    return {
        "buy_side_liquidity": round(float(buy_pools[0]["price"]), 5) if buy_pools else None,
        "sell_side_liquidity": round(float(sell_pools[0]["price"]), 5) if sell_pools else None,
        "pools": {
            "equal_highs": [round(float(p["price"]), 5) for p in eqh if p["type"] == "swing_high"],
            "equal_lows": [round(float(p["price"]), 5) for p in eqh if p["type"] == "swing_low"],
        },
    }


def filter_lit_sweeps(
    df: pd.DataFrame,
    sweeps: list[dict[str, Any]],
    swings: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=DEFAULT_CONFIG.atr_period)
    atr_last = float(atr.iloc[-1])
    tol = atr_last * DEFAULT_CONFIG.swing_alignment_atr

    pools = _find_equal_levels(swings) + _major_swing_extremes(swings)
    pool_prices = [(float(p["price"]), p["type"]) for p in pools]

    validated: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for sweep in sweeps:
        if not sweep.get("real_sweep") or not sweep.get("has_displacement"):
            continue

        bar_index = int(sweep.get("bar_index", -1))
        key = (bar_index, sweep.get("type", ""))
        if key in seen:
            continue

        liq_price = float(sweep.get("liquidity_price", sweep.get("price", 0)))
        if not any(abs(liq_price - price) <= tol for price, _ in pool_prices):
            continue

        item = dict(sweep)
        item["liquidity_price"] = round(liq_price, 5)
        if zones:
            nearest = min(range(len(zones)), key=lambda i: abs(zones[i]["center"] - liq_price))
            item["zone_index"] = nearest
            item["linked_zone_center"] = zones[nearest]["center"]
        seen.add(key)
        validated.append(item)

    return validated


def build_lit_scenarios(
    regime: dict[str, Any],
    sweeps: list[dict[str, Any]],
    bos_timeline: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    pools: dict[str, Any],
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    if sweeps and zones:
        latest = sorted(sweeps, key=lambda s: s.get("bar_index", 0))[-1]
        zone = zones[int(latest.get("zone_index", 0))]
        is_long = latest["type"] == "sell_side_sweep"
        scenarios.append(
            {
                "name": "Scenario 1",
                "setup_type": "liquidity sweep + rejection",
                "entry_logic": (
                    "Enter on retest of swept pool after displacement rejection candle"
                    if is_long
                    else "Enter short on retest after buy-side sweep rejection"
                ),
                "entry_zone": {"low": zone["low"], "high": zone["high"]},
                "invalidation": (
                    "Close below swept low / zone low"
                    if is_long
                    else "Close above swept high / zone high"
                ),
                "target_liquidity": (
                    pools.get("buy_side_liquidity") if is_long else pools.get("sell_side_liquidity")
                ),
            }
        )

    bos_events = [e for e in bos_timeline if e.get("type") == "BOS"]
    if bos_events and zones:
        last_bos = bos_events[-1]
        retest = min(zones, key=lambda z: abs(z["center"] - float(last_bos["level"])))
        is_bull = last_bos["direction"] == "bullish"
        scenarios.append(
            {
                "name": "Scenario 2",
                "setup_type": "break & retest BOS",
                "entry_logic": (
                    "Enter on first retest of broken structure level with bullish rejection"
                    if is_bull
                    else "Enter on retest of broken structure with bearish rejection"
                ),
                "entry_zone": {"low": retest["low"], "high": retest["high"]},
                "invalidation": (
                    "Close back below BOS level" if is_bull else "Close back above BOS level"
                ),
                "target_liquidity": (
                    pools.get("buy_side_liquidity") if is_bull else pools.get("sell_side_liquidity")
                ),
            }
        )

    if regime.get("regime") == "range" and scenarios:
        for scenario in scenarios:
            scenario["bias_note"] = "Two-sided liquidity hunt — no directional bias"

    return scenarios[:2]


def format_lit_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "low": round(float(z["low"]), 5),
            "high": round(float(z["high"]), 5),
            "center": round(float(z.get("center", z["mid"])), 5),
            "mid": round(float(z.get("mid", z["center"])), 5),
            "strength": int(z.get("strength", 0)),
            "type": z.get("type", "macro"),
            "valid_interactions": int(z.get("valid_interactions", 0)),
            "real_sweep_count": int(z.get("real_sweep_count", 0)),
            "source": z.get("source", "macro_swing"),
        }
        for z in zones
    ]


def calibrate_lit_output(
    df: pd.DataFrame,
    swings: list[dict[str, Any]],
    grouped_raw: dict[str, list[dict[str, Any]]],
    sweeps: list[dict[str, Any]],
    bos_timeline: list[dict[str, Any]],
    timeframe: str | None = None,
) -> dict[str, Any]:
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=DEFAULT_CONFIG.atr_period)
    last_close = float(data["close"].iloc[-1])

    bos_clean = calibrate_bos_choch(bos_timeline)
    regime = classify_lit_regime(bos_clean)
    lit_zones = build_lit_zones(data, grouped_raw, swings, sweeps, atr, bos_clean, timeframe)
    lit_sweeps = filter_lit_sweeps(data, sweeps, swings, lit_zones)
    pools = liquidity_pools_from_structure(swings, last_close)
    scenarios = build_lit_scenarios(regime, lit_sweeps, bos_clean, lit_zones, pools)

    return {
        "market_summary": {
            "current_price": round(last_close, 5),
            "regime": regime["regime"],
            "bias": regime["bias"],
        },
        "liquidity_targets": {
            "buy_side_liquidity": pools["buy_side_liquidity"],
            "sell_side_liquidity": pools["sell_side_liquidity"],
        },
        "liquidity_pools": pools["pools"],
        "zones": format_lit_zones(lit_zones),
        "zones_grouped": {
            "macro": [z for z in format_lit_zones(lit_zones) if z["type"] == "macro"],
            "mid": [z for z in format_lit_zones(lit_zones) if z["type"] == "mid"],
            "micro": [],
        },
        "sweeps": lit_sweeps,
        "bos_choch": bos_clean,
        "trade_scenarios": scenarios,
        "validation": {
            "zones_output": len(lit_zones),
            "sweeps_output": len(lit_sweeps),
            "notes": [
                "LIT institutional filter: max 5 zones, merge >50%, min 3 rejections + sweep.",
                f"Regime={regime['regime']} | Bias={regime['bias']} (BOS-sequence only).",
            ],
        },
    }
