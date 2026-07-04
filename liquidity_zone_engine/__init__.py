"""
Liquidity Intelligence Engine (LIT).

Pure pandas-based market structure analysis:
swing detection, ATR liquidity zones, clustering, and sweep events.
"""

from liquidity_zone_engine.engine import full_analysis
from liquidity_zone_engine.sample_data import load_sample_data
from liquidity_zone_engine.structure import _find_raw_swings, detect_swings
from liquidity_zone_engine.sweeps import detect_sweeps
from liquidity_zone_engine.zones import build_zones, flatten_zone_map
from liquidity_zone_engine.utils import normalize_ohlc

__all__ = [
    "build_zones",
    "detect_swings",
    "detect_sweeps",
    "full_analysis",
    "load_sample_data",
    "test_engine",
]


def test_engine():
    """Validate LIT-calibrated output (hard rules)."""
    df = load_sample_data()
    data = normalize_ohlc(df)

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    raw_swings = _find_raw_swings(data, highs, lows, swing_window=3)

    result = full_analysis(df, timeframe="H1")
    zones = result.get("zones_flat") or flatten_zone_map(result["zones"])
    sweeps = result["sweeps"]
    summary = result["market_summary"]

    assert len(result["swings"]) < len(raw_swings), "swing noise filter should reduce swing count"
    assert len(zones) <= 5, "H1 LIT output must have max 5 zones"
    assert summary["regime"] in ("trending_bull", "trending_bear", "range")
    if summary["regime"] == "range":
        assert summary["bias"] == "neutral"

    strengths = [z["strength"] for z in zones]
    if strengths:
        above_80 = sum(1 for s in strengths if s > 80)
        assert above_80 <= max(1, int(len(strengths) * 0.15) + 1)

    for sweep in sweeps:
        assert sweep.get("real_sweep") is True
        assert sweep.get("has_displacement")

    assert len(result["trade_scenarios"]) <= 2
    assert "liquidity_targets" in result
    assert result["liquidity_targets"]["buy_side_liquidity"] is None or isinstance(
        result["liquidity_targets"]["buy_side_liquidity"], (int, float)
    )

    return result
