"""
Liquidity-based trade entry trigger engine.

Converts liquidity_zone_engine analysis output (zones, swings, sweeps)
into structured entry signals. Execution logic only — no market analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

ZONE_K_DEFAULT = 0.6
MIN_ZONE_STRENGTH = 60
MIN_TOUCH_COUNT = 2
MIN_DISPLACEMENT_ATR = 1.2
SL_ATR_BUFFER = 0.2
MIN_RR = 2.0


def _zones_from_analysis(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    zones = analysis.get("zones", [])
    if isinstance(zones, dict):
        from liquidity_zone_engine.zones import flatten_zone_map

        return flatten_zone_map(zones)
    return zones


def _estimate_atr(zone: dict[str, Any]) -> float:
    height = float(zone["high"]) - float(zone["low"])
    if height <= 0:
        return 0.0
    return height / (2.0 * ZONE_K_DEFAULT)


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _structure_bias(swings: list[dict[str, Any]]) -> str:
    """Derive short-term structure from recent swings in analysis output."""
    if len(swings) < 2:
        return "neutral"

    ordered = sorted(swings, key=lambda item: _parse_time(item["timestamp"]))
    recent = ordered[-4:]

    highs = [item for item in recent if item["type"] == "swing_high"]
    lows = [item for item in recent if item["type"] == "swing_low"]

    higher_highs = len(highs) >= 2 and highs[-1]["price"] > highs[-2]["price"]
    higher_lows = len(lows) >= 2 and lows[-1]["price"] > lows[-2]["price"]
    lower_highs = len(highs) >= 2 and highs[-1]["price"] < highs[-2]["price"]
    lower_lows = len(lows) >= 2 and lows[-1]["price"] < lows[-2]["price"]

    if higher_highs and higher_lows:
        return "bullish"
    if lower_highs and lower_lows:
        return "bearish"
    return "neutral"


def filter_valid_sweeps(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep sweeps tied to strong zones with FVG POI and 3-candle confirmation."""
    zones = _zones_from_analysis(analysis)
    sweeps = analysis.get("sweeps", [])

    valid: list[dict[str, Any]] = []
    for sweep in sweeps:
        if sweep.get("significance") == "noise":
            continue
        if sweep.get("zone_level") == "micro":
            continue

        zone_index = sweep.get("zone_index")
        if zone_index is None or zone_index >= len(zones):
            continue

        zone = zones[zone_index]
        strength = int(zone.get("strength", 0))
        touch_count = int(zone.get("touch_count", 0))

        if strength < MIN_ZONE_STRENGTH:
            continue
        if touch_count < MIN_TOUCH_COUNT:
            continue
        if not sweep.get("three_candle_confirmed", False):
            continue
        if not sweep.get("fvg_poi_valid", False):
            continue

        valid.append({**sweep, "_zone": zone, "_zone_index": zone_index})

    return valid


def calculate_displacement(
    sweep: dict[str, Any],
    zone: dict[str, Any],
    swings: list[dict[str, Any]],
    setup_type: str,
) -> tuple[bool, float]:
    """
    Confirm post-sweep impulse using swings and zone geometry from analysis.

    Returns (is_valid, normalized_strength 0–1).
    """
    atr = _estimate_atr(zone)
    if atr <= 0:
        return False, 0.0

    sweep_time = _parse_time(sweep["timestamp"])
    swings_after = [
        item
        for item in swings
        if _parse_time(item["timestamp"]) > sweep_time
    ]

    zone_low = float(zone["low"])
    zone_high = float(zone["high"])
    sweep_price = float(sweep["price"])

    if setup_type == "buy":
        reference = min(zone_low, sweep_price)
        if swings_after:
            impulse_extreme = max(float(item["price"]) for item in swings_after)
        else:
            impulse_extreme = float(zone["center"])
        displacement = impulse_extreme - reference
    else:
        reference = max(zone_high, sweep_price)
        if swings_after:
            impulse_extreme = min(float(item["price"]) for item in swings_after)
        else:
            impulse_extreme = float(zone["center"])
        displacement = reference - impulse_extreme

    required = atr * MIN_DISPLACEMENT_ATR
    if displacement < required:
        return False, 0.0

    strength = min(1.0, displacement / (required * 1.5))
    return True, strength


def _sweep_quality_score(sweep: dict[str, Any], zone: dict[str, Any]) -> float:
    zone_low = float(zone["low"])
    zone_high = float(zone["high"])
    price = float(sweep["price"])
    zone_height = max(zone_high - zone_low, 1e-9)

    if sweep["type"] == "sell_side_sweep":
        penetration = max(0.0, zone_low - price)
    else:
        penetration = max(0.0, price - zone_high)

    penetration_ratio = min(1.0, penetration / (zone_height * 0.5))
    sweep_history = min(1.0, int(zone.get("sweep_count", 0)) / 3.0)
    fvg_bonus = 0.15 if sweep.get("fvg_poi_valid") else 0.0
    candle_bonus = 0.15 if sweep.get("three_candle_confirmed") else 0.0
    return min(1.0, 0.5 * penetration_ratio + 0.35 * sweep_history + fvg_bonus + candle_bonus)


def _structure_alignment_score(setup_type: str, structure: str) -> float:
    if structure == "neutral":
        return 1.0
    if setup_type == "buy" and structure == "bullish":
        return 1.0
    if setup_type == "sell" and structure == "bearish":
        return 1.0
    if setup_type == "buy" and structure == "bearish":
        return 0.35
    if setup_type == "sell" and structure == "bullish":
        return 0.35
    return 0.7


def _session_alignment_score(setup_type: str, session: dict[str, Any]) -> float:
    reversal = session.get("reversal_bias")
    if not session.get("asian_sweep_active") or reversal is None:
        return 0.85
    if setup_type == reversal:
        return 1.0
    return 0.45


def _daily_bias_alignment_score(setup_type: str, daily_bias: dict[str, Any]) -> float:
    bias = daily_bias.get("bias", "neutral")
    if bias == "neutral":
        return 0.85
    if setup_type == "buy" and bias == "bullish":
        return 1.0
    if setup_type == "sell" and bias == "bearish":
        return 1.0
    if setup_type == "buy" and bias == "bearish":
        return 0.4
    if setup_type == "sell" and bias == "bullish":
        return 0.4
    return 0.7


def compute_confidence(
    sweep: dict[str, Any],
    zone: dict[str, Any],
    displacement_strength: float,
    setup_type: str,
    structure: str,
    session: dict[str, Any],
    daily_bias: dict[str, Any],
) -> int:
    sweep_quality = _sweep_quality_score(sweep, zone)
    zone_strength = min(1.0, int(zone.get("strength", 0)) / 100.0)
    structure_score = _structure_alignment_score(setup_type, structure)
    session_score = _session_alignment_score(setup_type, session)
    bias_score = _daily_bias_alignment_score(setup_type, daily_bias)

    raw = (
        sweep_quality * 25.0
        + displacement_strength * 25.0
        + zone_strength * 20.0
        + structure_score * 15.0
        + session_score * 10.0
        + bias_score * 5.0
    )
    return min(100, max(0, int(round(raw))))


def _next_opposing_zone(
    zones: list[dict[str, Any]],
    zone_index: int,
    setup_type: str,
) -> float | None:
    current = zones[zone_index]
    current_center = float(current["center"])

    if setup_type == "buy":
        candidates = [
            float(item["center"])
            for idx, item in enumerate(zones)
            if idx != zone_index and float(item["center"]) > current_center
        ]
        return min(candidates) if candidates else None

    candidates = [
        float(item["center"])
        for idx, item in enumerate(zones)
        if idx != zone_index and float(item["center"]) < current_center
    ]
    return max(candidates) if candidates else None


def build_trade_plan(
    sweep: dict[str, Any],
    zone: dict[str, Any],
    zone_index: int,
    zones: list[dict[str, Any]],
    setup_type: str,
    confidence: int,
    displacement_strength: float,
    structure: str,
    session: dict[str, Any],
    daily_bias: dict[str, Any],
) -> dict[str, Any]:
    atr = _estimate_atr(zone)
    zone_low = float(zone["low"])
    zone_high = float(zone["high"])
    sweep_price = float(sweep["price"])
    sweep_time = _parse_time(sweep["timestamp"])
    fvg_poi = sweep.get("fvg_poi")

    if setup_type == "buy":
        entry_price = float(fvg_poi["low"]) if fvg_poi else zone_low
        sweep_extreme = min(zone_low, sweep_price)
        stop_loss = sweep_extreme - atr * SL_ATR_BUFFER
        risk = entry_price - stop_loss
        opposing = _next_opposing_zone(zones, zone_index, setup_type)
        tp_from_zone = opposing if opposing is not None else entry_price + risk * MIN_RR
        take_profit = max(tp_from_zone, entry_price + risk * MIN_RR)
        reason = (
            "Sell-side sweep + FVG POI retest + 3-candle confirmation; "
            f"structure={structure}, session_reversal={session.get('reversal_bias')}, "
            f"daily_bias={daily_bias.get('bias')}"
        )
    else:
        entry_price = float(fvg_poi["high"]) if fvg_poi else zone_high
        sweep_extreme = max(zone_high, sweep_price)
        stop_loss = sweep_extreme + atr * SL_ATR_BUFFER
        risk = stop_loss - entry_price
        opposing = _next_opposing_zone(zones, zone_index, setup_type)
        tp_from_zone = opposing if opposing is not None else entry_price - risk * MIN_RR
        take_profit = min(tp_from_zone, entry_price - risk * MIN_RR)
        reason = (
            "Buy-side sweep + FVG POI retest + 3-candle confirmation; "
            f"structure={structure}, session_reversal={session.get('reversal_bias')}, "
            f"daily_bias={daily_bias.get('bias')}"
        )

    if risk <= 0:
        risk = atr * 0.5
        if setup_type == "buy":
            take_profit = entry_price + risk * MIN_RR
        else:
            take_profit = entry_price - risk * MIN_RR

    return {
        "type": setup_type,
        "entry_price": round(entry_price, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit": round(take_profit, 5),
        "confidence": confidence,
        "reason": reason,
        "related_zone": zone_index,
        "time": sweep_time,
        "_displacement_strength": displacement_strength,
    }


def detect_entries(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Build actionable entry signals from liquidity analysis output.

    Expects the dict returned by liquidity_zone_engine.full_analysis().
    """
    zones = _zones_from_analysis(analysis)
    swings = analysis.get("swings", [])
    session = analysis.get("session", {})
    daily_bias = analysis.get("daily_bias", {})
    structure = _structure_bias(swings)

    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for sweep in filter_valid_sweeps(analysis):
        zone = sweep["_zone"]
        zone_index = sweep["_zone_index"]

        if sweep["type"] == "sell_side_sweep":
            setup_type = "buy"
        elif sweep["type"] == "buy_side_sweep":
            setup_type = "sell"
        else:
            continue

        if session.get("asian_sweep_active") and session.get("reversal_bias"):
            if setup_type != session["reversal_bias"]:
                continue

        dedupe_key = (zone_index, setup_type)
        if dedupe_key in seen:
            continue

        displacement_ok, displacement_strength = calculate_displacement(
            sweep, zone, swings, setup_type
        )
        if not displacement_ok:
            continue

        confidence = compute_confidence(
            sweep,
            zone,
            displacement_strength,
            setup_type,
            structure,
            session,
            daily_bias,
        )
        if confidence < 45:
            continue

        plan = build_trade_plan(
            sweep=sweep,
            zone=zone,
            zone_index=zone_index,
            zones=zones,
            setup_type=setup_type,
            confidence=confidence,
            displacement_strength=displacement_strength,
            structure=structure,
            session=session,
            daily_bias=daily_bias,
        )
        plan.pop("_displacement_strength", None)
        entries.append(plan)
        seen.add(dedupe_key)

    entries.sort(key=lambda item: item["time"])
    return {"entries": entries}


def test_entry_engine():
    """Smoke test using sample data and full liquidity analysis."""
    from liquidity_zone_engine import full_analysis, load_sample_data

    df = load_sample_data()
    analysis = full_analysis(df)
    return detect_entries(analysis)
