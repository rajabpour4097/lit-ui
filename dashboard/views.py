"""Liquidity dashboard views."""

from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone

from dashboard.data_access import get_ohlc_data
from liquidity_zone_engine import full_analysis
from liquidity_zone_engine.zones import flatten_zone_map

SWING_TYPE_FA = {
    "swing_high": "سقف سویینگ",
    "swing_low": "کف سویینگ",
}

SWEEP_TYPE_FA = {
    "buy_side_sweep": "جاروب سمت خرید",
    "sell_side_sweep": "جاروب سمت فروش",
}

SWEEP_CLASS_FA = {
    "buy_side_liquidity_grab": "جذب نقدینگی سمت خرید",
    "sell_side_liquidity_grab": "جذب نقدینگی سمت فروش",
}

ZONE_TYPE_FA = {
    "High": "عرضه (High)",
    "Low": "تقاضا (Low)",
    "Equilibrium": "تعادل",
}

LEVEL_FA = {
    "macro": "ماکرو",
    "mid": "میانی",
    "micro": "ریز",
}

TREND_FA = {
    "bullish": "صعودی",
    "bearish": "نزولی",
    "neutral": "خنثی",
    "range": "رنج",
    "trending_bull": "روند صعودی",
    "trending_bear": "روند نزولی",
}


def _format_timestamp(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(str(value))
    if timezone.is_naive(ts):
        ts = timezone.make_aware(ts, dt_timezone.utc)
    local_ts = timezone.localtime(ts)
    return local_ts.strftime("%Y/%m/%d %H:%M")


def _zones_from_result(result: dict) -> list:
    """Support grouped or legacy flat zone payloads."""
    zones = result.get("zones", [])
    if isinstance(zones, dict):
        return flatten_zone_map(zones)
    return zones


def _format_price(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        value = value.get("center") or value.get("price")
    if value is None:
        return "—"
    return f"{float(value):.5f}"


def _prepare_display(result: dict) -> dict:
    """Add Persian display labels for validated SMC dashboard output."""
    targets = result.get("liquidity_targets", {})
    buy_target = targets.get("buy_side_liquidity")
    sell_target = targets.get("sell_side_liquidity")

    flat_zones = result.get("zones_flat")
    if flat_zones:
        zone_list = flat_zones
    else:
        zone_list = _zones_from_result(result)

    market_summary = result.get("market_summary", {})
    current_price = market_summary.get("current_price", 0)

    if buy_target is None and zone_list:
        above = sorted(
            [z for z in zone_list if float(z.get("center", z.get("mid", 0))) > current_price],
            key=lambda z: float(z.get("center", z.get("mid", 0))),
        )
        if above:
            buy_target = float(above[0].get("center", above[0].get("mid")))

    if sell_target is None and zone_list:
        below = sorted(
            [z for z in zone_list if float(z.get("center", z.get("mid", 0))) < current_price],
            key=lambda z: float(z.get("center", z.get("mid", 0))),
            reverse=True,
        )
        if below:
            sell_target = float(below[0].get("center", below[0].get("mid")))

    zones = []
    for zone in zone_list:
        zone_type = zone.get("zone_type") or zone.get("type", "Equilibrium")
        mid = zone.get("mid", zone.get("center"))
        zones.append(
            {
                **zone,
                "low_fmt": _format_price(zone.get("low")),
                "high_fmt": _format_price(zone.get("high")),
                "center_fmt": _format_price(mid),
                "zone_type_fa": ZONE_TYPE_FA.get(str(zone_type), str(zone_type)),
                "level_fa": LEVEL_FA.get(zone.get("level", ""), zone.get("level", "—")),
            }
        )

    sweeps = []
    for sweep in result.get("sweeps", []):
        classification = sweep.get("classification", "")
        sweeps.append(
            {
                **sweep,
                "type_fa": SWEEP_TYPE_FA.get(sweep["type"], sweep["type"]),
                "classification_fa": SWEEP_CLASS_FA.get(classification, classification or "—"),
                "price_fmt": f"{sweep['price']:.5f}",
                "score": sweep.get("score", 0),
                "timestamp_fmt": _format_timestamp(sweep.get("timestamp")),
                "linked_zone_fmt": _format_price(sweep.get("linked_zone_center")),
            }
        )

    swings = []
    for swing in result.get("swings", []):
        swings.append(
            {
                **swing,
                "type_fa": SWING_TYPE_FA.get(swing["type"], swing["type"]),
                "price_fmt": f"{swing['price']:.5f}",
                "timestamp_fmt": _format_timestamp(swing.get("timestamp")),
            }
        )

    max_strength = max((zone["strength"] for zone in zones), default=1)
    validation = result.get("validation", {})
    trend = market_summary.get("bias", "neutral")

    return {
        "zones": zones,
        "sweeps": sweeps,
        "swings": swings,
        "zone_count": len(zones),
        "swing_count": len(swings),
        "sweep_count": len(sweeps),
        "max_strength": max_strength,
        "market_summary": market_summary,
        "market_structure": {
            "trend": trend,
            "trend_fa": TREND_FA.get(trend, trend),
            "regime": market_summary.get("regime", "range"),
            "regime_fa": TREND_FA.get(
                market_summary.get("regime", "range"),
                market_summary.get("regime", "range"),
            ),
            "last_close": market_summary.get("current_price"),
            "buy_target_fmt": _format_price(buy_target),
            "sell_target_fmt": _format_price(sell_target),
            "key_bos_levels": result.get("bos_choch", [])[-4:],
        },
        "bos_choch": result.get("bos_choch", []),
        "trade_scenarios": result.get("trade_scenarios", []),
        "validation": validation,
        "zones_before": validation.get("zones_before", len(zones)),
        "zones_after": validation.get("zones_after", len(zones)),
        "sweeps_before": validation.get("sweeps_before", len(sweeps)),
        "sweeps_after": validation.get("sweeps_after", len(sweeps)),
        "fix_notes": validation.get("notes", []),
    }


def liquidity_dashboard(request: HttpRequest) -> HttpResponse:
    symbol = request.GET.get("symbol", "EURUSD")
    timeframe = request.GET.get("timeframe", "H1")

    symbols = getattr(settings, "LIQUIDITY_SYMBOLS", ["EURUSD", "XAUUSD", "GBPUSD"])
    timeframes = getattr(settings, "LIQUIDITY_TIMEFRAMES", ["D1", "H4", "H1", "M15", "M5"])

    if symbol not in symbols:
        symbol = symbols[0]
    if timeframe not in timeframes:
        timeframe = "H1"

    error_message = None
    data = None
    display = None

    try:
        df = get_ohlc_data(symbol, timeframe)
        data = full_analysis(df, timeframe=timeframe)
        display = _prepare_display(data)
    except Exception as exc:
        error_message = f"خطا در دریافت یا تحلیل داده: {exc}"

    return render(
        request,
        "liquidity/dashboard.html",
        {
            "data": data,
            "display": display,
            "symbol": symbol,
            "timeframe": timeframe,
            "symbols": symbols,
            "timeframes": timeframes,
            "error_message": error_message,
        },
    )
