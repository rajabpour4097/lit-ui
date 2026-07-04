"""Synthetic OHLC sample data for offline testing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_sample_data(bars: int = 120) -> pd.DataFrame:
    """Build deterministic OHLC data with visible swings and sweep scenarios."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-01-01", periods=bars, freq="15min", tz="UTC")

    price = 2000.0
    rows: list[dict[str, float]] = []

    for i in range(bars):
        cycle = i % 24

        if cycle in (6, 7):
            drift = 2.5
            volatility = 1.2
        elif cycle in (12, 13):
            drift = -2.5
            volatility = 1.2
        elif cycle in (18, 19):
            drift = 0.0
            volatility = 2.5
        else:
            drift = rng.uniform(-0.4, 0.4)
            volatility = 0.8

        open_price = price
        close_price = open_price + drift + rng.uniform(-volatility, volatility)
        high_price = max(open_price, close_price) + abs(rng.uniform(0.1, volatility))
        low_price = min(open_price, close_price) - abs(rng.uniform(0.1, volatility))

        if i == 80:
            high_price = open_price + 8.0
            close_price = open_price + 1.0
        if i == 95:
            low_price = open_price - 8.0
            close_price = open_price - 1.0

        rows.append(
            {
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
            }
        )
        price = close_price

    df = pd.DataFrame(rows, index=timestamps)
    df["timestamp"] = df.index
    return df
