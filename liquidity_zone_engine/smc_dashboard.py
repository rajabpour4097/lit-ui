"""SMC Institutional Trading Engine — less detection, more decision."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.atr import compute_atr
from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc, resolve_timestamp


def filter_controlled_swings(
    raw_swings: list[dict[str, Any]],
    data: pd.DataFrame,
    atr: pd.Series,
) -> list[dict[str, Any]]:
    """
    Keep significant structure swings only.

    Merge same-type swings closer than max(0.15% price, 0.3*ATR).
    Cap at smc_max_swings (default 20).
    """
    config = DEFAULT_CONFIG
    if not raw_swings:
        return []

    kept: list[dict[str, Any]] = []
    for swing in raw_swings:
        idx = int(swing["index"])
        price = float(swing["price"])
        atr_val = float(atr.iloc[idx])
        threshold = max(price * config.smc_swing_min_pct, atr_val * config.smc_swing_min_atr)

        if not kept:
            kept.append(swing)
            continue

        prev = kept[-1]
        prev_price = float(prev["price"])

        if swing["type"] == prev["type"] and abs(price - prev_price) < threshold:
            if swing["type"] == "swing_high" and price >= prev_price:
                kept[-1] = swing
            elif swing["type"] == "swing_low" and price <= prev_price:
                kept[-1] = swing
            continue

        if abs(price - prev_price) < threshold:
            continue

        kept.append(swing)

    return kept[-config.smc_max_swings :]


def _cluster_swings_atr(
    swings: list[dict[str, Any]],
    atr: pd.Series,
    cluster_mult: float,
) -> list[list[dict[str, Any]]]:
    if not swings:
        return []

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [swings[0]]

    for swing in swings[1:]:
        if swing["type"] != current[0]["type"]:
            clusters.append(current)
            current = [swing]
            continue

        idx = int(swing["index"])
        threshold = float(atr.iloc[idx]) * cluster_mult
        if abs(float(swing["price"]) - float(current[-1]["price"])) <= threshold:
            current.append(swing)
        else:
            clusters.append(current)
            current = [swing]

    clusters.append(current)
    return clusters


def _count_touches(data: pd.DataFrame, zone: dict[str, Any]) -> int:
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    return sum(
        1 for i in range(len(data)) if lows[i] <= zone["high"] and highs[i] >= zone["low"]
    )


def _bos_reactions(zone: dict[str, Any], bos_timeline: list[dict[str, Any]]) -> int:
    return sum(
        1
        for e in bos_timeline
        if zone["low"] <= float(e["level"]) <= zone["high"]
    )


def _zone_from_cluster(
    cluster: list[dict[str, Any]],
    level: str,
    data: pd.DataFrame,
    bos_timeline: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    atr: pd.Series,
) -> dict[str, Any]:
    prices = [float(s["price"]) for s in cluster]
    low, high = min(prices), max(prices)
    center = sum(prices) / len(prices)
    idx = int(cluster[-1]["index"])
    tol = float(atr.iloc[idx]) * DEFAULT_CONFIG.smc_zone_cluster_atr

    sweep_confirmed = any(
        abs(float(s.get("liquidity_price", 0)) - center) <= tol and s.get("confirmed")
        for s in sweeps
    )
    touch_count = _count_touches(data, {"low": low, "high": high})
    bos_count = _bos_reactions({"low": low, "high": high}, bos_timeline)

    raw = touch_count * 20 + bos_count * 30 + (50 if sweep_confirmed else 0)
    return {
        "low": low,
        "high": high,
        "center": center,
        "mid": center,
        "type": level,
        "level": level,
        "swing_type": cluster[0]["type"],
        "touch_count": touch_count,
        "bos_reactions": bos_count,
        "sweep_confirmed": sweep_confirmed,
        "cluster_size": len(cluster),
        "_raw_strength": raw,
    }


def _normalize_strengths(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not zones:
        return []
    max_raw = max(float(z.get("_raw_strength", 0)) for z in zones) or 1.0
    return [
        {
            **zone,
            "strength": int(min(100, max(5, round(float(zone.get("_raw_strength", 0)) / max_raw * 100)))),
        }
        for zone in zones
    ]


def _assign_tier(cluster: list[dict[str, Any]], atr: pd.Series) -> str:
    idx = int(cluster[-1]["index"])
    span = max(float(s["price"]) for s in cluster) - min(float(s["price"]) for s in cluster)
    atr_val = float(atr.iloc[idx])
    size = len(cluster)

    if size >= 3 or span >= atr_val * 1.2:
        return "macro"
    if size >= 2 or span >= atr_val * 0.5:
        return "mid"
    return "micro"


def build_smc_zones(
    df: pd.DataFrame,
    swings: list[dict[str, Any]],
    bos_timeline: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Cluster swings by 0.15*ATR; cap macro 4 / mid 5 / micro 3 (max 12 total)."""
    config = DEFAULT_CONFIG
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=config.atr_period)

    if not swings:
        return {"macro": [], "mid": [], "micro": []}

    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]

    all_clusters: list[tuple[str, list[dict[str, Any]]]] = []
    for side in (highs, lows):
        for cluster in _cluster_swings_atr(side, atr, config.smc_zone_cluster_atr):
            tier = _assign_tier(cluster, atr)
            all_clusters.append((tier, cluster))

    buckets: dict[str, list[dict[str, Any]]] = {"macro": [], "mid": [], "micro": []}
    for tier, cluster in all_clusters:
        zone = _zone_from_cluster(cluster, tier, data, bos_timeline, sweeps, atr)
        buckets[tier].append(zone)

    for tier in buckets:
        buckets[tier].sort(key=lambda z: z["_raw_strength"], reverse=True)

    macro = _normalize_strengths(buckets["macro"][: config.macro_max_zones])
    mid = _normalize_strengths(buckets["mid"][: config.mid_max_zones])
    micro = _normalize_strengths(buckets["micro"][: config.micro_max_zones])

    combined = macro + mid + micro
    if len(combined) > config.smc_max_zones_total:
        combined.sort(key=lambda z: z["_raw_strength"], reverse=True)
        combined = combined[: config.smc_max_zones_total]
        macro, mid, micro = [], [], []
        caps = {
            "macro": config.macro_max_zones,
            "mid": config.mid_max_zones,
            "micro": config.micro_max_zones,
        }
        counts = {"macro": 0, "mid": 0, "micro": 0}
        for zone in combined:
            tier = zone["level"]
            if counts[tier] < caps[tier]:
                if tier == "macro":
                    macro.append(zone)
                elif tier == "mid":
                    mid.append(zone)
                else:
                    micro.append(zone)
                counts[tier] += 1

    return {"macro": macro, "mid": mid, "micro": micro}


def classify_regime_from_bos(bos_timeline: list[dict[str, Any]]) -> str:
    """Regime from last 5 BOS/CHOCH events only."""
    window = bos_timeline[-DEFAULT_CONFIG.smc_bos_window :]
    bos_only = [e for e in window if e.get("type") == "BOS"]

    if len(bos_only) < 2:
        if len(window) >= 2:
            dirs = [e["direction"] for e in window]
            if dirs.count("bullish") >= 2 and dirs.count("bearish") == 0:
                return "trending_bull"
            if dirs.count("bearish") >= 2 and dirs.count("bullish") == 0:
                return "trending_bear"
        return "range"

    consecutive_bull = 0
    consecutive_bear = 0
    max_bull = 0
    max_bear = 0
    for event in bos_only:
        if event["direction"] == "bullish":
            consecutive_bull += 1
            consecutive_bear = 0
        else:
            consecutive_bear += 1
            consecutive_bull = 0
        max_bull = max(max_bull, consecutive_bull)
        max_bear = max(max_bear, consecutive_bear)

    choch_in_window = sum(1 for e in window if e.get("type") == "CHOCH")

    if max_bull >= 2:
        return "trending_bull"
    if max_bear >= 2:
        return "trending_bear"
    if choch_in_window >= 1:
        return "range"
    return "range"


def decide_trading_bias(
    last_close: float,
    targets: dict[str, Any],
    sweeps: list[dict[str, Any]],
    atr_last: float,
) -> str:
    """
    Bias from liquidity proximity + valid sweep.

    No valid sweep → neutral (WAIT).
    """
    config = DEFAULT_CONFIG
    if not sweeps:
        return "neutral"

    buy_liq = targets.get("buy_side_liquidity")
    sell_liq = targets.get("sell_side_liquidity")
    prox = max(last_close * config.smc_liquidity_proximity_pct, atr_last * 0.5)

    recent = sorted(
        [s for s in sweeps if s.get("confirmed")],
        key=lambda s: s.get("bar_index", 0),
    )
    if not recent:
        return "neutral"

    latest = recent[-1]

    if (
        buy_liq is not None
        and abs(last_close - float(buy_liq)) <= prox
        and latest["type"] == "sell_side_sweep"
    ):
        return "bearish"

    if (
        sell_liq is not None
        and abs(last_close - float(sell_liq)) <= prox
        and latest["type"] == "buy_side_sweep"
    ):
        return "bullish"

    if latest["type"] == "sell_side_sweep":
        return "bullish"
    if latest["type"] == "buy_side_sweep":
        return "bearish"

    return "neutral"


def liquidity_targets_from_swings(
    swings: list[dict[str, Any]],
    last_close: float,
) -> dict[str, Any]:
    highs = sorted(
        float(s["price"])
        for s in swings
        if s["type"] == "swing_high" and float(s["price"]) > last_close
    )
    lows = sorted(
        (float(s["price"]) for s in swings if s["type"] == "swing_low" and float(s["price"]) < last_close),
        reverse=True,
    )
    return {
        "buy_side_liquidity": round(highs[0], 5) if highs else None,
        "sell_side_liquidity": round(lows[0], 5) if lows else None,
    }


def _next_opposite_zone(
    zones: list[dict[str, Any]],
    from_price: float,
    direction: str,
) -> float | None:
    if direction == "long":
        candidates = [z for z in zones if z["center"] > from_price]
        if candidates:
            return round(min(candidates, key=lambda z: z["center"])["center"], 5)
    else:
        candidates = [z for z in zones if z["center"] < from_price]
        if candidates:
            return round(max(candidates, key=lambda z: z["center"])["center"], 5)
    return None


def build_smc_scenarios(
    data: pd.DataFrame,
    sweeps: list[dict[str, Any]],
    bos_timeline: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    targets: dict[str, Any],
    atr_last: float,
) -> list[dict[str, Any]]:
    """Max 2 scenarios with entry / SL / TP / one-line reasoning."""
    config = DEFAULT_CONFIG
    last_close = float(data["close"].iloc[-1])
    scenarios: list[dict[str, Any]] = []

    confirmed = [s for s in sweeps if s.get("confirmed")]
    latest_sweep = confirmed[-1] if confirmed else None

    if latest_sweep:
        bar = int(latest_sweep["bar_index"])
        high = float(data["high"].iloc[bar])
        low = float(data["low"].iloc[bar])
        sweep_mid = (high + low) / 2.0
        is_long = latest_sweep["type"] == "sell_side_sweep"
        liq = float(latest_sweep["liquidity_price"])
        entry = round(sweep_mid, 5)
        sl = round(
            (liq - atr_last * config.smc_sl_atr_buffer) if is_long else (liq + atr_last * config.smc_sl_atr_buffer),
            5,
        )
        tp = (
            _next_opposite_zone(zones, last_close, "long")
            or targets.get("buy_side_liquidity")
            if is_long
            else _next_opposite_zone(zones, last_close, "short")
            or targets.get("sell_side_liquidity")
        )
        scenarios.append(
            {
                "name": "Scenario A",
                "setup_type": "Sweep Reversal",
                "active": True,
                "entry": entry,
                "entry_zone": {"low": round(min(entry, sweep_mid - atr_last * 0.1), 5), "high": round(max(entry, sweep_mid + atr_last * 0.1), 5)},
                "stop_loss": sl,
                "take_profit": tp,
                "reasoning": (
                    "Sell-side sweep rejected — long at 50% sweep candle range"
                    if is_long
                    else "Buy-side sweep rejected — short at 50% sweep candle range"
                ),
            }
        )
    else:
        scenarios.append(
            {
                "name": "Scenario A",
                "setup_type": "Sweep Reversal",
                "active": False,
                "entry": None,
                "entry_zone": None,
                "stop_loss": None,
                "take_profit": targets.get("buy_side_liquidity") or targets.get("sell_side_liquidity"),
                "reasoning": "No confirmed sweep — WAIT for liquidity grab",
            }
        )

    bos_events = [e for e in bos_timeline if e.get("type") == "BOS"]
    last_bos = bos_events[-1] if bos_events else None

    if last_bos and zones:
        level = float(last_bos["level"])
        retest = min(zones, key=lambda z: abs(z["center"] - level))
        is_bull = last_bos["direction"] == "bullish"
        entry = round(retest["center"], 5)
        sl = round(
            (level - atr_last * config.smc_sl_atr_buffer) if is_bull else (level + atr_last * config.smc_sl_atr_buffer),
            5,
        )
        tp = (
            targets.get("buy_side_liquidity")
            if is_bull
            else targets.get("sell_side_liquidity")
        )
        scenarios.append(
            {
                "name": "Scenario B",
                "setup_type": "BOS Retest",
                "active": True,
                "entry": entry,
                "entry_zone": {"low": retest["low"], "high": retest["high"]},
                "stop_loss": sl,
                "take_profit": tp,
                "reasoning": (
                    f"Bullish BOS retest at {round(level, 5)} — enter on structure hold"
                    if is_bull
                    else f"Bearish BOS retest at {round(level, 5)} — enter on structure hold"
                ),
            }
        )
    else:
        fallback = zones[0] if zones else None
        scenarios.append(
            {
                "name": "Scenario B",
                "setup_type": "BOS Retest",
                "active": False,
                "entry": round(fallback["center"], 5) if fallback else None,
                "entry_zone": (
                    {"low": fallback["low"], "high": fallback["high"]} if fallback else None
                ),
                "stop_loss": None,
                "take_profit": targets.get("sell_side_liquidity") or targets.get("buy_side_liquidity"),
                "reasoning": "No confirmed BOS retest — WAIT for structure break",
            }
        )

    return scenarios[:2]


def filter_smc_sweeps(sweeps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep confirmed sweeps only; cap output; dedupe by bar."""
    config = DEFAULT_CONFIG
    valid = [s for s in sweeps if s.get("confirmed")]
    by_bar: dict[tuple[int, str], dict[str, Any]] = {}
    for sweep in valid:
        key = (int(sweep.get("bar_index", -1)), sweep.get("type", ""))
        if key not in by_bar or sweep.get("score", 0) > by_bar[key].get("score", 0):
            by_bar[key] = sweep

    deduped = sorted(by_bar.values(), key=lambda s: s.get("bar_index", 0))
    return deduped[-config.smc_max_sweeps_output :]


def format_smc_zones(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for level in ("macro", "mid", "micro"):
        for zone in grouped.get(level, []):
            flat.append(
                {
                    "low": round(float(zone["low"]), 5),
                    "high": round(float(zone["high"]), 5),
                    "center": round(float(zone["center"]), 5),
                    "mid": round(float(zone.get("mid", zone["center"])), 5),
                    "strength": int(zone.get("strength", 0)),
                    "type": level,
                    "level": level,
                    "touch_count": int(zone.get("touch_count", 0)),
                    "sweep_confirmed": bool(zone.get("sweep_confirmed", False)),
                }
            )
    return flat


def calibrate_smc_output(
    df: pd.DataFrame,
    swings: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    bos_timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    data = normalize_ohlc(df)
    atr = compute_atr(data, period=DEFAULT_CONFIG.atr_period)
    last_close = float(data["close"].iloc[-1])
    atr_last = float(atr.iloc[-1])

    bos_last5 = bos_timeline[-DEFAULT_CONFIG.smc_bos_window :]
    regime = classify_regime_from_bos(bos_timeline)
    targets = liquidity_targets_from_swings(swings, last_close)

    grouped = build_smc_zones(data, swings, bos_timeline, sweeps)
    flat = format_smc_zones(grouped)

    sweeps_out = filter_smc_sweeps(sweeps)
    bias = decide_trading_bias(last_close, targets, sweeps_out, atr_last)

    scenarios = build_smc_scenarios(data, sweeps_out, bos_last5, flat, targets, atr_last)

    return {
        "market_summary": {
            "current_price": round(last_close, 5),
            "regime": regime,
            "bias": bias,
            "action": "WAIT" if bias == "neutral" else "TRADE",
            "buy_side_liquidity": targets["buy_side_liquidity"],
            "sell_side_liquidity": targets["sell_side_liquidity"],
        },
        "liquidity_targets": targets,
        "liquidity_pools": {"equal_highs": [], "equal_lows": []},
        "zones": flat,
        "zones_grouped": grouped,
        "sweeps": sweeps_out,
        "bos_choch": bos_last5,
        "trade_scenarios": scenarios,
        "validation": {
            "zones_output": len(flat),
            "sweeps_output": len(sweeps_out),
            "swings_input": len(swings),
            "bos_window": len(bos_last5),
            "notes": [
                f"SMC institutional: max {DEFAULT_CONFIG.smc_max_zones_total} zones, max {DEFAULT_CONFIG.smc_max_sweeps_output} sweeps.",
                f"Regime={regime} | Bias={bias} (last {DEFAULT_CONFIG.smc_bos_window} BOS + sweep engine).",
            ],
        },
    }
