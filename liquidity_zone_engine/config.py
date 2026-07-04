"""Default configuration for the liquidity zone engine."""

from dataclasses import dataclass

SUPPORTED_TIMEFRAMES = ("D1", "H4", "H1", "M15", "M5")

REQUIRED_OHLC_COLUMNS = ("open", "high", "low", "close")

HTF_DOMINANCE = ("D1", "H4", "H1", "M15", "M5")


@dataclass(frozen=True)
class EngineConfig:
    swing_window: int = 3
    atr_period: int = 14
    zone_k: float = 0.6
    cluster_atr_mult: float = 0.3

    min_swing_displacement_atr: float = 0.8
    consolidation_atr: float = 0.8
    impulse_displacement_atr: float = 1.2
    sweep_displacement_atr: float = 0.35
    max_zone_height_atr: float = 3.0

    zone_merge_atr: float = 1.4
    zone_overlap_merge_pct: float = 0.5
    min_zone_strength: int = 15

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
