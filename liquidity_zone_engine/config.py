"""Default configuration for the liquidity zone engine."""

from dataclasses import dataclass

SUPPORTED_TIMEFRAMES = ("D1", "H4", "H1", "M15", "M5")

REQUIRED_OHLC_COLUMNS = ("open", "high", "low", "close")

HTF_DOMINANCE = ("D1", "H4", "H1", "M15", "M5")

# Broker server wall clock (Turkey / UTC+3 on this account).
BROKER_UTC_OFFSET_HOURS = 3

# Soft stale threshold (market closed / overnight) — analysis still runs.
BROKER_STALE_BAR_SECONDS = 6 * 3600

# Hard reject only when MT5 data is clearly unusable.
BROKER_MAX_BAR_AGE_SECONDS = 30 * 86400

# Asian liquidity = Sydney + Tokyo overlap on broker clock.
ASIAN_SESSION_BROKER_START_HOUR = 1
ASIAN_SESSION_BROKER_END_HOUR = 10


@dataclass(frozen=True)
class MarketSession:
    key: str
    label: str
    region_timezone: str
    open_hour: int
    close_hour: int


# Market-center hours as shown in the MT5 session table (broker UTC+3).
FOREX_MARKET_SESSIONS = (
    MarketSession("sydney", "Sydney", "Australia/Sydney", 1, 9),
    MarketSession("tokyo", "Tokyo", "Asia/Tokyo", 2, 10),
    MarketSession("frankfurt", "Frankfurt", "Europe/Berlin", 9, 17),
    MarketSession("london", "London", "Europe/London", 10, 18),
    MarketSession("new_york", "New York", "America/New_York", 15, 23),
)

ZONE_LEVELS = {"macro": 1, "mid": 2, "micro": 3}


@dataclass(frozen=True)
class EngineConfig:
    swing_window: int = 3
    atr_period: int = 14
    zone_k: float = 0.6
    cluster_atr_mult: float = 0.3

    min_swing_displacement_atr: float = 0.8
    consolidation_atr: float = 0.8
    impulse_displacement_atr: float = 1.2
    macro_swing_atr: float = 1.5
    mid_swing_atr: float = 0.8
    same_level_merge_atr: float = 0.6
    sweep_displacement_atr: float = 0.35
    sweep_wick_body_ratio: float = 1.5
    max_zone_height_atr: float = 3.0

    min_zone_touches: int = 2
    min_zone_sweeps: int = 1
    max_inactive_bars: int = 40
    cluster_min_points: int = 2
    micro_cluster_min_points: int = 1

    lit_max_zones_h1: int = 5
    lit_max_zones_default: int = 8
    lit_zone_merge_overlap: float = 0.5
    lit_min_zone_interactions: int = 3
    lit_strength_top_pct: float = 0.10

    bos_break_atr: float = 0.8
    bos_revert_bars: int = 3
    sweep_reclaim_bars: int = 1
    sweep_min_displacement_atr: float = 0.5
    lit_swing_post_displacement_atr: float = 0.8
    macro_max_zones: int = 5
    macro_min_zones: int = 3
    macro_merge_overlap: float = 0.6
    mid_max_zones: int = 12
    mid_min_zones: int = 5
    micro_max_zones: int = 30
    micro_min_zones: int = 5
    min_total_zones: int = 10

    min_sweep_score: int = 50
    sweep_score_wick: int = 30
    sweep_score_volume: int = 20
    sweep_score_macro: int = 20
    sweep_score_displacement: int = 10

    regime_range_macro_pct: float = 0.6

    validation_merge_atr: float = 0.15
    min_institutional_strength: int = 35
    swing_alignment_atr: float = 0.5

    bos_confirm_atr: float = 0.25
    bos_fake_retrace_bars: int = 3
    fvg_min_gap_atr: float = 0.15
    micro_equal_level_atr: float = 0.25

    strength_weight_rejection: int = 3
    strength_weight_sweep: int = 4
    strength_weight_displacement: int = 5
    strength_weight_time: int = 1

    def __post_init__(self) -> None:
        if self.swing_window < 1:
            raise ValueError("swing_window must be >= 1")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if not 0.4 <= self.zone_k <= 0.8:
            raise ValueError("zone_k must be between 0.4 and 0.8")
        if self.cluster_atr_mult <= 0:
            raise ValueError("cluster_atr_mult must be > 0")


DEFAULT_CONFIG = EngineConfig()
