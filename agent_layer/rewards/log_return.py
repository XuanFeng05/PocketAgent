from __future__ import annotations

import math


def net_asset_log_return(nav_before: float, nav_after: float) -> float:
    """Return the additive log growth of net asset value after all costs."""
    if nav_before <= 0 or nav_after <= 0:
        raise ValueError("Net asset values must remain positive.")
    return math.log(float(nav_after) / float(nav_before))
