"""OHLC data access for the liquidity dashboard."""

from __future__ import annotations

import pandas as pd
from django.conf import settings

TIMEFRAME_TO_MT5 = {
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def get_ohlc_data(symbol: str, timeframe: str, bars: int | None = None) -> pd.DataFrame:
    """
    Fetch OHLC data for analysis.

    Uses the existing MT5 terminal connection (read-only). Does not modify
    any connector or authentication code in the trading bot layer.
    """
    if timeframe not in TIMEFRAME_TO_MT5:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    bar_count = bars or getattr(settings, "OHLC_BAR_COUNT", 500)

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is not installed.") from exc

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        tf_minutes = TIMEFRAME_TO_MT5[timeframe]
        mt5_tf = _resolve_mt5_timeframe(mt5, tf_minutes)

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bar_count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No OHLC data for {symbol} {timeframe}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df[["open", "high", "low", "close", "timestamp"]].copy()
    finally:
        mt5.shutdown()


def _resolve_mt5_timeframe(mt5, minutes: int):
    mapping = {
        5: mt5.TIMEFRAME_M5,
        15: mt5.TIMEFRAME_M15,
        60: mt5.TIMEFRAME_H1,
        240: mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
    }
    if minutes not in mapping:
        raise ValueError(f"Unsupported MT5 timeframe minutes: {minutes}")
    return mapping[minutes]
