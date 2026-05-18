"""Parse the gradient-update horizon out of a Diff-MPPI planner name.

Conventions used by `benchmark_diff_mppi.cu`:
- ``diff_mppi_3_early1``  -> horizon 1 step
- ``diff_mppi_3_early2``  -> horizon 2
- ``diff_mppi_3_early4``  -> horizon 4
- ``diff_mppi_3_early8``  -> horizon 8
- ``diff_mppi_3_early16`` -> horizon 16
- ``diff_mppi_3``         -> full horizon (sentinel 0 in the C++ side; we
                            surface it as 0 here so callers can spot it)
- ``mppi``, ``feedback_*``, ``step_mppi``                -> no gradient horizon (None)

The convention is intentionally narrow: callers that need to recover the
actual full-horizon length (the C++ default is 30 steps) must keep that
constant themselves.
"""

from __future__ import annotations


FULL_HORIZON_SENTINEL = 0


def parse_grad_update_horizon(planner: str) -> int | None:
    if not planner.startswith("diff_mppi"):
        return None
    if "_early" in planner:
        suffix = planner.split("_early", 1)[1]
        try:
            return int(suffix)
        except ValueError:
            return None
    return FULL_HORIZON_SENTINEL
