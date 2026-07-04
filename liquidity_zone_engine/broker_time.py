"""Broker timezone validation and forex session windows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from liquidity_zone_engine.config import (
    ASIAN_SESSION_BROKER_END_HOUR,
    ASIAN_SESSION_BROKER_START_HOUR,
    BROKER_MAX_BAR_AGE_SECONDS,
    BROKER_STALE_BAR_SECONDS,
    BROKER_UTC_OFFSET_HOURS,
    FOREX_MARKET_SESSIONS,
)


def broker_timezone_label(offset_hours: int | None = None) -> str:
    hours = BROKER_UTC_OFFSET_HOURS if offset_hours is None else offset_hours
    sign = "+" if hours >= 0 else "-"
    return f"{sign}{abs(hours):02d}:00"


def broker_tzinfo(offset_hours: int | None = None) -> timezone:
    hours = BROKER_UTC_OFFSET_HOURS if offset_hours is None else offset_hours
    return timezone(timedelta(hours=hours))


def normalize_timestamps_utc(data: pd.DataFrame) -> pd.Series:
    """Return UTC-aware timestamps from a normalized OHLC dataframe."""
    if "timestamp" in data.columns:
        ts = pd.to_datetime(data["timestamp"], utc=True)
    elif isinstance(data.index, pd.DatetimeIndex):
        idx = data.index
        ts = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    else:
        raise ValueError("DataFrame requires timestamp column or DatetimeIndex")

    return pd.Series(ts, index=data.index)


def to_broker_time(
    timestamps: pd.Series,
    offset_hours: int | None = None,
) -> pd.Series:
    """Convert UTC timestamps to broker wall-clock time."""
    hours = BROKER_UTC_OFFSET_HOURS if offset_hours is None else offset_hours
    utc = pd.to_datetime(timestamps, utc=True)
    return utc.dt.tz_convert(broker_tzinfo(hours))


def stamp_broker_metadata(
    df: pd.DataFrame,
    *,
    offset_hours: int | None = None,
    validated: bool = False,
    verification: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach broker timezone metadata used by validation before analysis."""
    hours = BROKER_UTC_OFFSET_HOURS if offset_hours is None else offset_hours
    df = df.copy()
    df.attrs["broker_utc_offset_hours"] = hours
    df.attrs["broker_timezone"] = broker_timezone_label(hours)
    df.attrs["broker_timezone_validated"] = validated
    if verification:
        df.attrs["bar_age_seconds"] = verification.get("bar_age_seconds")
        df.attrs["data_stale"] = verification.get("data_stale", False)
    return df


def _build_broker_verification(
    *,
    expected: int,
    symbol: str,
    tick_utc: datetime | None,
    bar_utc: datetime,
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    bar_age_seconds = max(0, int((now_utc - bar_utc).total_seconds()))
    tick_age_seconds = None
    if tick_utc is not None:
        tick_age_seconds = max(0, int((now_utc - tick_utc).total_seconds()))

    if bar_age_seconds > BROKER_MAX_BAR_AGE_SECONDS:
        raise RuntimeError(
            f"Latest MT5 bar is too old ({bar_age_seconds}s) — "
            f"maximum allowed age is {BROKER_MAX_BAR_AGE_SECONDS}s"
        )

    data_stale = bar_age_seconds > BROKER_STALE_BAR_SECONDS
    result: dict[str, Any] = {
        "broker_timezone": broker_timezone_label(expected),
        "broker_offset_hours": expected,
        "timezone_validated": True,
        "symbol": symbol,
        "bar_age_seconds": bar_age_seconds,
        "data_stale": data_stale,
        "last_bar_utc": bar_utc.isoformat(),
        "last_bar_broker": bar_utc.astimezone(broker_tzinfo(expected)).isoformat(),
    }
    if tick_utc is not None:
        result["tick_utc"] = tick_utc.isoformat()
        result["tick_broker"] = tick_utc.astimezone(broker_tzinfo(expected)).isoformat()
        result["tick_age_seconds"] = tick_age_seconds
    return result


def verify_mt5_broker_offset_session(
    mt5: Any,
    symbol: str,
    *,
    last_bar_utc: datetime,
    expected_offset_hours: int | None = None,
) -> dict[str, Any]:
    """
    Validate broker offset using an already-open MT5 session.

    Does not reject stale bars while the market is closed; only flags
    ``data_stale`` so session logic can still run on the last available day.
    """
    expected = BROKER_UTC_OFFSET_HOURS if expected_offset_hours is None else expected_offset_hours

    if last_bar_utc.tzinfo is None:
        last_bar_utc = last_bar_utc.replace(tzinfo=timezone.utc)
    else:
        last_bar_utc = last_bar_utc.astimezone(timezone.utc)

    tick_utc: datetime | None = None
    tick = mt5.symbol_info_tick(symbol)
    if tick is not None and int(getattr(tick, "time", 0)) > 0:
        tick_utc = datetime.fromtimestamp(int(tick.time), tz=timezone.utc)

    return _build_broker_verification(
        expected=expected,
        symbol=symbol,
        tick_utc=tick_utc,
        bar_utc=last_bar_utc,
    )


def validate_broker_timezone(
    df: pd.DataFrame,
    *,
    expected_offset_hours: int | None = None,
    require_metadata: bool = False,
) -> dict[str, Any]:
    """
    Validate broker timezone before every analysis run.

    Raises ValueError when the dataframe metadata or timestamps disagree with
    the configured broker offset (+03:00 by default).
    """
    expected = BROKER_UTC_OFFSET_HOURS if expected_offset_hours is None else expected_offset_hours
    data = df.copy()
    if data.empty:
        raise ValueError("Cannot validate broker timezone on empty OHLC data")

    if require_metadata and "broker_utc_offset_hours" not in df.attrs:
        raise ValueError(
            "OHLC data missing broker timezone metadata. "
            "Load data through get_ohlc_data() or stamp_broker_metadata()."
        )

    declared = df.attrs.get("broker_utc_offset_hours")
    if declared is not None and int(declared) != expected:
        raise ValueError(
            f"Broker offset mismatch: data declares UTC{broker_timezone_label(int(declared))}, "
            f"engine expects UTC{broker_timezone_label(expected)}"
        )

    timestamps = normalize_timestamps_utc(data)
    if timestamps.isna().any():
        raise ValueError("Timestamps contain null values — broker timezone check failed")

    last_ts = pd.Timestamp(timestamps.iloc[-1])
    if last_ts.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware (UTC) for session analysis")

    result = {
        "broker_timezone": broker_timezone_label(expected),
        "broker_offset_hours": expected,
        "timezone_validated": True,
        "last_bar_utc": last_ts.isoformat(),
        "last_bar_broker": to_broker_time(pd.Series([last_ts]), expected).iloc[0].isoformat(),
    }
    if "data_stale" in df.attrs:
        result["data_stale"] = bool(df.attrs["data_stale"])
    if "bar_age_seconds" in df.attrs:
        result["bar_age_seconds"] = int(df.attrs["bar_age_seconds"])
    return result


def verify_mt5_broker_offset(
    symbol: str,
    *,
    expected_offset_hours: int | None = None,
    last_bar_utc: datetime | None = None,
) -> dict[str, Any]:
    """
    Confirm MT5 is connected and broker wall-clock offset matches configuration.

    Stale bars (market closed / weekend) are allowed; ``data_stale=True`` is
    returned instead of raising when the latest bar is older than 6 hours.
    """
    expected = BROKER_UTC_OFFSET_HOURS if expected_offset_hours is None else expected_offset_hours

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is not installed.") from exc

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        if last_bar_utc is None:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"No MT5 rates for {symbol}: {mt5.last_error()}")
            last_bar_utc = datetime.fromtimestamp(int(rates[0]["time"]), tz=timezone.utc)

        return verify_mt5_broker_offset_session(
            mt5,
            symbol,
            last_bar_utc=last_bar_utc,
            expected_offset_hours=expected,
        )
    finally:
        mt5.shutdown()


def _session_hour_mask(hours: pd.Series, open_hour: int, close_hour: int) -> pd.Series:
    if open_hour == close_hour:
        return hours == open_hour
    if open_hour < close_hour:
        return (hours >= open_hour) & (hours < close_hour)
    return (hours >= open_hour) | (hours < close_hour)


def _session_range(
    day_data: pd.DataFrame,
    mask: pd.Series,
) -> dict[str, float] | None:
    segment = day_data.loc[mask]
    if segment.empty:
        return None
    return {
        "high": round(float(segment["high"].max()), 5),
        "low": round(float(segment["low"].min()), 5),
    }


def _active_sessions(broker_hour: int, offset_hours: int) -> list[str]:
    active: list[str] = []
    for session in FOREX_MARKET_SESSIONS:
        mask = _session_hour_mask(
            pd.Series([broker_hour]),
            session.open_hour,
            session.close_hour,
        )
        if bool(mask.iloc[0]):
            active.append(session.key)
    return active


def build_market_session_snapshot(
    data: pd.DataFrame,
    timestamps_utc: pd.Series,
    *,
    offset_hours: int | None = None,
) -> dict[str, Any]:
    """Build per-market session status using broker wall-clock hours."""
    hours_offset = BROKER_UTC_OFFSET_HOURS if offset_hours is None else offset_hours
    broker_ts = to_broker_time(timestamps_utc, hours_offset)

    latest_day = broker_ts.iloc[-1].date()
    day_mask = broker_ts.dt.date == latest_day
    day_data = data.loc[day_mask]
    day_broker = broker_ts.loc[day_mask]
    day_hours = day_broker.dt.hour

    if day_data.empty:
        day_data = data
        day_hours = broker_ts.dt.hour
        latest_day = broker_ts.iloc[-1].date()

    broker_hour = int(day_broker.iloc[-1].hour) if not day_broker.empty else int(broker_ts.iloc[-1].hour)
    markets: dict[str, Any] = {}

    for session in FOREX_MARKET_SESSIONS:
        mask = _session_hour_mask(day_hours, session.open_hour, session.close_hour)
        session_range = _session_range(day_data, mask)
        markets[session.key] = {
            "label": session.label,
            "region_timezone": session.region_timezone,
            "open_broker": f"{session.open_hour:02d}:00",
            "close_broker": f"{session.close_hour:02d}:00",
            "active": session.key in _active_sessions(broker_hour, hours_offset),
            "range": session_range,
        }

    return {
        "date_broker": str(latest_day),
        "broker_hour": broker_hour,
        "markets": markets,
        "active_sessions": _active_sessions(broker_hour, hours_offset),
    }


def asian_session_mask(broker_hours: pd.Series) -> pd.Series:
    """Asian liquidity window = Sydney/Tokyo overlap on broker clock (01:00–10:00)."""
    return (broker_hours >= ASIAN_SESSION_BROKER_START_HOUR) & (
        broker_hours < ASIAN_SESSION_BROKER_END_HOUR
    )


def ny_midnight_open_price(
    data: pd.DataFrame,
    timestamps_utc: pd.Series,
) -> float:
    """First bar at/after America/New_York midnight mapped onto UTC timestamps."""
    ny_tz = ZoneInfo("America/New_York")
    utc_ts = pd.to_datetime(timestamps_utc, utc=True)
    ny_latest = utc_ts.iloc[-1].tz_convert(ny_tz)
    ny_day = ny_latest.date()

    ny_day_start = datetime.combine(ny_day, datetime.min.time(), tzinfo=ny_tz)
    ny_midnight_utc = ny_day_start.astimezone(timezone.utc)

    ny_mask = utc_ts >= pd.Timestamp(ny_midnight_utc)
    ny_bars = data.loc[ny_mask]
    if ny_bars.empty:
        return float(data["open"].iloc[0])
    return float(ny_bars["open"].iloc[0])
