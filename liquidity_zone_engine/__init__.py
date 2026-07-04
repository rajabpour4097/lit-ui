"""
Liquidity Intelligence Engine (LIT / SMC).

Pure pandas-based market structure analysis:
swing detection, liquidity zones, clustering, and sweep events.
"""

from liquidity_zone_engine.engine import full_analysis
from liquidity_zone_engine.sample_data import load_sample_data
from liquidity_zone_engine.structure import _find_raw_swings, detect_swings
from liquidity_zone_engine.sweeps import detect_sweeps
from liquidity_zone_engine.zones import build_zones, flatten_zone_map
from liquidity_zone_engine.utils import normalize_ohlc
from liquidity_zone_engine.config import DEFAULT_CONFIG

__all__ = [
    "build_zones",
    "detect_swings",
    "detect_sweeps",
    "full_analysis",
    "load_sample_data",
    "test_engine",
]


def test_engine():
    """Validate SMC institutional output (controlled detection)."""
    df = load_sample_data()
    result = full_analysis(df, timeframe="H1")
    zones = result.get("zones_flat") or flatten_zone_map(result["zones"])
    sweeps = result["sweeps"]
    summary = result["market_summary"]
    swings = result["swings"]

    assert len(swings) <= DEFAULT_CONFIG.smc_max_swings
    assert len(swings) >= 1
    assert len(zones) <= DEFAULT_CONFIG.smc_max_zones_total
    assert summary["regime"] in ("trending_bull", "trending_bear", "range")
    assert len(result["bos_choch"]) <= DEFAULT_CONFIG.smc_bos_window
    assert len(result["trade_scenarios"]) == 2
    assert "liquidity_targets" in result

    for sweep in sweeps:
        assert sweep.get("confirmed") is True
        assert sweep.get("score", 0) >= 60

    for scenario in result["trade_scenarios"]:
        assert "reasoning" in scenario
        if scenario.get("active"):
            assert scenario.get("stop_loss") is not None
            assert scenario.get("take_profit") is not None

    return result
