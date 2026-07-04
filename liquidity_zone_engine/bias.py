"""Daily bias and AMD cycle helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from liquidity_zone_engine.fvg import detect_fvgs
from liquidity_zone_engine.sessions import _series_timestamps
from liquidity_zone_engine.utils import normalize_ohlc


def compute_daily_bias(df: pd.DataFrame, fvgs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Derive daily bias using AMD logic and prior-day FVG fill behavior.

    If a previous-day FVG is filled, bias flips opposite to the fill candle direction.
    """
    data = normalize_ohlc(df)
    timestamps = _series_timestamps(data)
    all_fvgs = fvgs if fvgs is not None else detect_fvgs(df)

    if data.empty:
        return {"bias": "neutral", "phase": "unknown", "reason": "empty data"}

    latest_date = timestamps.iloc[-1].date()
    prior_fvgs = [
        fvg
        for fvg in all_fvgs
        if pd.Timestamp(timestamps.iloc[fvg["index"]]).date() < latest_date
    ]

    bias = "neutral"
    reason = "no prior-day FVG fill signal"
    phase = _estimate_amd_phase(data, timestamps)

    ny_open_bias = None
    for fvg in reversed(prior_fvgs):
        if not fvg.get("filled"):
            continue
        fill_index = fvg.get("fill_index")
        if fill_index is None:
            continue

        fill_date = timestamps.iloc[fill_index].date()
        if fill_date != latest_date:
            continue

        open_price = float(data["open"].iloc[fill_index])
        close_price = float(data["close"].iloc[fill_index])

        if fvg["type"] == "bullish":
            bias = "bearish" if close_price < open_price else "bullish"
            reason = "prior-day bullish FVG filled — bias flipped by fill candle"
        else:
            bias = "bullish" if close_price > open_price else "bearish"
            reason = "prior-day bearish FVG filled — bias flipped by fill candle"
        break

    return {
        "bias": bias,
        "phase": phase,
        "reason": reason,
        "ny_open_bias": ny_open_bias,
    }


def _estimate_amd_phase(data: pd.DataFrame, timestamps: pd.Series) -> str:
    """Rough AMD phase from broker session progression on the latest day."""
    from liquidity_zone_engine.broker_time import to_broker_time

    broker_ts = to_broker_time(timestamps)
    latest_day = broker_ts.iloc[-1].date()
    day_mask = broker_ts.dt.date == latest_day
    day_hours = broker_ts.loc[day_mask].dt.hour

    if day_hours.empty:
        return "unknown"

    hour = int(day_hours.iloc[-1])
    if 1 <= hour < 10:
        return "accumulation"
    if 10 <= hour < 15:
        return "manipulation"
    if 15 <= hour < 23:
        return "distribution"
    return "accumulation"
