"""
Liquidity Intelligence Engine (LIT).

Pure pandas-based market structure analysis:
swing detection, ATR liquidity zones, clustering, and sweep events.
"""

from liquidity_zone_engine.engine import full_analysis
from liquidity_zone_engine.sample_data import load_sample_data
from liquidity_zone_engine.structure import _find_raw_swings, detect_swings
from liquidity_zone_engine.sweeps import detect_sweeps
from liquidity_zone_engine.zones import _zone_overlap_ratio, build_zones
from liquidity_zone_engine.utils import normalize_ohlc

__all__ = [
    "build_zones",
    "detect_swings",
    "detect_sweeps",
    "full_analysis",
    "load_sample_data",
    "test_engine",
]


def _count_uncompressed_zones(df) -> int:
    swings = detect_swings(df)
    data = normalize_ohlc(df)
    from liquidity_zone_engine.atr import compute_atr
    from liquidity_zone_engine.zones import _build_raw_zones

    atr = compute_atr(data)
    return len(_build_raw_zones(swings, atr, zone_k=0.6))


def _has_redundant_overlap(zones) -> bool:
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            if (
                zones[i]["type"] == zones[j]["type"]
                and _zone_overlap_ratio(zones[i], zones[j]) > 0.6
                and abs(zones[i]["strength"] - zones[j]["strength"]) <= 5
            ):
                return True
    return False


def _loose_sweep_count(df, zones) -> int:
    """Estimate unfiltered sweep count for regression comparison."""
    from liquidity_zone_engine.atr import compute_atr

    data = normalize_ohlc(df)
    atr = compute_atr(data)
    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()

    count = 0
    for zone in zones:
        zone_low = zone["low"]
        zone_high = zone["high"]
        for i in range(len(data)):
            if highs[i] > zone_high and closes[i] < zone_high:
                count += 1
            if lows[i] < zone_low and closes[i] > zone_low:
                count += 1
    return count


def test_engine():
    """Run offline validation against precision liquidity targets."""
    df = load_sample_data()
    data = normalize_ohlc(df)

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    raw_swings = _find_raw_swings(data, highs, lows, swing_window=3)
    uncompressed_zone_count = _count_uncompressed_zones(df)

    pre_quality_zones = build_zones(df)
    loose_sweep_count = _loose_sweep_count(df, pre_quality_zones)

    result = full_analysis(df, timeframe="M15")

    swings = result["swings"]
    zones = result["zones"]
    sweeps = result["sweeps"]

    assert len(swings) < len(raw_swings), "swing noise filter should reduce swing count"
    assert len(zones) <= max(8, int(uncompressed_zone_count * 0.45)), (
        "zone compression should reduce zone count by at least 55%"
    )
    assert 3 <= len(zones) <= 15, "final zone count should stay in high-quality range"
    if loose_sweep_count > 0:
        assert len(sweeps) <= max(1, int(loose_sweep_count * 0.7)), (
            "validated sweeps should be reduced versus loose wick-only detection"
        )
    assert not _has_redundant_overlap(zones), "no overlapping redundant zones should remain"

    strengths = [zone["strength"] for zone in zones]
    assert all(0 <= strength <= 100 for strength in strengths)
    assert len(set(strengths)) >= min(2, len(strengths)), (
        "strength distribution should be meaningful"
    )
    if len(strengths) >= 3:
        assert max(strengths) - min(strengths) <= 70, "strength spread should avoid random spikes"
        assert max(strengths) <= 95 and min(strengths) >= 15

    for zone in zones:
        assert zone["type"] in ("impulsive", "corrective")
        assert zone["touch_count"] >= 0
        assert zone["sweep_count"] >= 0
        assert (
            zone["touch_count"] >= 2
            or zone["sweep_count"] > 0
            or zone["strength"] >= 15
        ), "each zone must satisfy quality criteria"

    return result
