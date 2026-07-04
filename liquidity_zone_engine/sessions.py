"""Session-based liquidity (Asian range, NY midnight open, market centers)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.broker_time import (
    asian_session_mask,
    build_market_session_snapshot,
    normalize_timestamps_utc,
    ny_midnight_open_price,
    to_broker_time,
    validate_broker_timezone,
)
from liquidity_zone_engine.utils import normalize_ohlc


def _series_timestamps(data: pd.DataFrame) -> pd.Series:
    return normalize_timestamps_utc(data)


def analyze_session_liquidity(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute Asian range, market-center status, and NY midnight open reference.

    Session windows use broker wall clock (UTC+03:00) matching the MT5
    market-center table (Sydney/Tokyo/Frankfurt/London/New York).
    """
    data = normalize_ohlc(df)
    tz_info = validate_broker_timezone(data)
    timestamps = _series_timestamps(data)
    broker_ts = to_broker_time(timestamps)

    latest_day = broker_ts.iloc[-1].date()
    day_mask = broker_ts.dt.date == latest_day
    day_data = data.loc[day_mask]
    day_broker = broker_ts.loc[day_mask]

    if day_data.empty:
        day_data = data
        day_broker = broker_ts
        latest_day = broker_ts.iloc[-1].date()

    day_hours = day_broker.dt.hour
    asian_mask = asian_session_mask(day_hours)
    asian_data = day_data.loc[asian_mask]

    if asian_data.empty:
        asian_high = float(day_data["high"].iloc[: max(1, len(day_data) // 3)].max())
        asian_low = float(day_data["low"].iloc[: max(1, len(day_data) // 3)].min())
    else:
        asian_high = float(asian_data["high"].max())
        asian_low = float(asian_data["low"].min())

    last_high = float(data["high"].iloc[-1])
    last_low = float(data["low"].iloc[-1])
    last_close = float(data["close"].iloc[-1])

    asian_high_swept = last_high > asian_high
    asian_low_swept = last_low < asian_low

    reversal_bias: str | None = None
    if asian_high_swept and not asian_low_swept:
        reversal_bias = "sell"
    elif asian_low_swept and not asian_high_swept:
        reversal_bias = "buy"
    elif asian_high_swept and asian_low_swept:
        mid = (asian_high + asian_low) / 2.0
        reversal_bias = "buy" if last_close < mid else "sell"

    ny_midnight_open = ny_midnight_open_price(data, timestamps)
    market_snapshot = build_market_session_snapshot(data, timestamps)

    payload: dict[str, Any] = {
        **tz_info,
        **market_snapshot,
        "asian_range": {
            "high": round(asian_high, 5),
            "low": round(asian_low, 5),
            "date": str(latest_day),
            "open_broker": f"{day_hours.min() if not day_hours.empty else 0:02d}:00",
            "window_broker": "01:00-10:00",
        },
        "asian_high_swept": asian_high_swept,
        "asian_low_swept": asian_low_swept,
        "asian_sweep_active": asian_high_swept or asian_low_swept,
        "reversal_bias": reversal_bias,
        "ny_midnight_open": round(ny_midnight_open, 5),
        "price_vs_ny_open": "above" if last_close >= ny_midnight_open else "below",
        "last_close": round(last_close, 5),
    }
    if tz_info.get("data_stale"):
        payload["data_stale"] = True
        payload["market_status"] = "closed_or_stale"
        payload["active_sessions"] = []
        for market in payload.get("markets", {}).values():
            market["active"] = False
    else:
        payload["market_status"] = "live"
    return payload
