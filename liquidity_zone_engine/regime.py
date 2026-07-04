"""Market regime classification and trade scenario generation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.config import DEFAULT_CONFIG
from liquidity_zone_engine.utils import normalize_ohlc


def _price_in_zone(close: float, zone: dict[str, Any]) -> bool:
    return zone["low"] <= close <= zone["high"]


def classify_market_regime(
    df: pd.DataFrame,
    macro_zones: list[dict[str, Any]],
    bos_timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    data = normalize_ohlc(df)
    closes = data["close"].to_numpy()
    last_close = float(closes[-1])

    if not macro_zones:
        return {"regime": "range", "bias": "neutral", "confidence": 0.4}

    primary_macro = max(macro_zones, key=lambda z: z.get("strength", 0))
    in_macro_bars = sum(
        1 for c in closes if primary_macro["low"] <= c <= primary_macro["high"]
    )
    macro_time_pct = in_macro_bars / max(len(closes), 1)

    real_bos = [e for e in bos_timeline if e.get("validity") == "confirmed"]
    bullish_bos = sum(1 for e in real_bos if e.get("direction") == "bullish")
    bearish_bos = sum(1 for e in real_bos if e.get("direction") == "bearish")

    if macro_time_pct >= DEFAULT_CONFIG.regime_range_macro_pct:
        regime = "range"
    elif bullish_bos >= 2 and bullish_bos > bearish_bos:
        regime = "trending_bull"
    elif bearish_bos >= 2 and bearish_bos > bullish_bos:
        regime = "trending_bear"
    elif _price_in_zone(last_close, primary_macro) and macro_time_pct > 0.45:
        mid = (primary_macro["high"] + primary_macro["low"]) / 2.0
        regime = "accumulation" if last_close <= mid else "distribution"
    else:
        regime = "range"

    if regime == "trending_bull":
        bias = "bullish"
    elif regime == "trending_bear":
        bias = "bearish"
    elif regime == "accumulation":
        bias = "bullish"
    elif regime == "distribution":
        bias = "bearish"
    else:
        bias = "neutral"

    confidence = min(0.95, 0.45 + macro_time_pct * 0.3 + len(real_bos) * 0.05)
    return {
        "regime": regime,
        "bias": bias,
        "confidence": round(confidence, 2),
        "macro_time_pct": round(macro_time_pct, 2),
    }


def build_trade_scenarios(
    regime: dict[str, Any],
    zones_flat: list[dict[str, Any]],
    sweeps: list[dict[str, Any]],
    last_close: float,
) -> list[dict[str, Any]]:
    if regime.get("regime") not in ("range", "trending_bull", "trending_bear", "accumulation", "distribution"):
        return []

    scenarios: list[dict[str, Any]] = []
    macro = [z for z in zones_flat if z.get("level") == "macro"]
    mid = [z for z in zones_flat if z.get("level") == "mid"]

    if sweeps:
        latest = sorted(sweeps, key=lambda s: s.get("bar_index", 0))[-1]
        zone_idx = latest.get("zone_index")
        linked = zones_flat[zone_idx] if zone_idx is not None and zone_idx < len(zones_flat) else None
        if linked:
            if latest["type"] == "sell_side_sweep":
                scenarios.append(
                    {
                        "name": "Scenario A",
                        "setup_type": "sweep + rejection (buy)",
                        "entry_zone": {"low": linked["low"], "high": linked["high"]},
                        "invalidation": round(linked["low"] - (linked["high"] - linked["low"]) * 0.5, 5),
                        "target_liquidity": _nearest_liquidity_above(zones_flat, last_close),
                    }
                )
            else:
                scenarios.append(
                    {
                        "name": "Scenario A",
                        "setup_type": "sweep + rejection (sell)",
                        "entry_zone": {"low": linked["low"], "high": linked["high"]},
                        "invalidation": round(linked["high"] + (linked["high"] - linked["low"]) * 0.5, 5),
                        "target_liquidity": _nearest_liquidity_below(zones_flat, last_close),
                    }
                )

    if regime.get("regime") == "range" and macro:
        z = macro[0]
        scenarios.append(
            {
                "name": "Scenario B",
                "setup_type": "range fade at macro boundary",
                "entry_zone": {"low": z["low"], "high": z["high"]},
                "invalidation": round(z["low"] - (z["high"] - z["low"]), 5),
                "target_liquidity": _nearest_liquidity_above(zones_flat, last_close),
            }
        )

    if regime.get("regime") in ("trending_bull", "trending_bear") and mid:
        z = mid[-1]
        direction = "buy" if regime["regime"] == "trending_bull" else "sell"
        scenarios.append(
            {
                "name": "Scenario B",
                "setup_type": f"BOS retest ({direction})",
                "entry_zone": {"low": z["low"], "high": z["high"]},
                "invalidation": round(z["low"] if direction == "buy" else z["high"], 5),
                "target_liquidity": (
                    _nearest_liquidity_above(zones_flat, last_close)
                    if direction == "buy"
                    else _nearest_liquidity_below(zones_flat, last_close)
                ),
            }
        )

    return scenarios


def _nearest_liquidity_above(zones: list[dict[str, Any]], price: float) -> float | None:
    above = [z["center"] for z in zones if z["center"] > price]
    return round(min(above), 5) if above else None


def _nearest_liquidity_below(zones: list[dict[str, Any]], price: float) -> float | None:
    below = [z["center"] for z in zones if z["center"] < price]
    return round(max(below), 5) if below else None


def build_market_summary(
    df: pd.DataFrame,
    regime: dict[str, Any],
) -> dict[str, Any]:
    data = normalize_ohlc(df)
    last_close = float(data["close"].iloc[-1])
    return {
        "current_price": round(last_close, 5),
        "regime": regime.get("regime", "range"),
        "bias": regime.get("bias", "neutral"),
        "confidence": regime.get("confidence", 0.5),
    }
