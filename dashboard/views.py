"""Liquidity dashboard views."""

from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone

from dashboard.data_access import get_ohlc_data
from liquidity_zone_engine import full_analysis

SWING_TYPE_FA = {
    "swing_high": "سقف سویینگ",
    "swing_low": "کف سویینگ",
}

SWEEP_TYPE_FA = {
    "buy_side_sweep": "جاروب سمت خرید",
    "sell_side_sweep": "جاروب سمت فروش",
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


def _prepare_display(result: dict) -> dict:
    """Add Persian display labels without recalculating analysis."""
    zones = []
    for zone in result.get("zones", []):
        zones.append(
            {
                **zone,
                "low_fmt": f"{zone['low']:.5f}",
                "high_fmt": f"{zone['high']:.5f}",
                "center_fmt": f"{zone['center']:.5f}",
            }
        )

    sweeps = []
    for sweep in result.get("sweeps", []):
        sweeps.append(
            {
                **sweep,
                "type_fa": SWEEP_TYPE_FA.get(sweep["type"], sweep["type"]),
                "price_fmt": f"{sweep['price']:.5f}",
                "timestamp_fmt": _format_timestamp(sweep.get("timestamp")),
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

    return {
        "zones": zones,
        "sweeps": sweeps,
        "swings": swings,
        "zone_count": len(zones),
        "swing_count": len(swings),
        "sweep_count": len(sweeps),
        "max_strength": max_strength,
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
