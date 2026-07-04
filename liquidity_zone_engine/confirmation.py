"""Three-candle swing formation confirmation."""

from __future__ import annotations

import pandas as pd


def three_candle_confirmation(
    data: pd.DataFrame,
    bar_index: int,
    setup_type: str,
) -> bool:
    """
    Confirm protected zone using 3-candle formation.

    Candle 2 (bar_index) must hold the extreme wick.
    Candle 3 must close beyond candle 2 body in the reversal direction.
    """
    if bar_index < 1 or bar_index + 1 >= len(data):
        return False

    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()

    c1_high = highs[bar_index - 1]
    c1_low = lows[bar_index - 1]
    c2_open = opens[bar_index]
    c2_close = closes[bar_index]
    c2_high = highs[bar_index]
    c2_low = lows[bar_index]
    c3_close = closes[bar_index + 1]

    c2_body_top = max(c2_open, c2_close)
    c2_body_bottom = min(c2_open, c2_close)

    if setup_type == "buy":
        lower_wick = c2_body_bottom - c2_low
        body = abs(c2_close - c2_open) or 1e-9
        extreme_wick = lower_wick >= body * 0.8 and c2_low <= c1_low
        return extreme_wick and c3_close > c2_body_top

    upper_wick = c2_high - c2_body_top
    body = abs(c2_close - c2_open) or 1e-9
    extreme_wick = upper_wick >= body * 0.8 and c2_high >= c1_high
    return extreme_wick and c3_close < c2_body_bottom
