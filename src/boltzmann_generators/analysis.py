"""Analysis helpers for weighted and unweighted population estimates."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from .services.analysis import AnalysisSuite

RegionFn = Callable[[Tensor], Tensor]

__all__ = ["AnalysisSuite", "RegionFn", "basin_populations", "rectangular_region"]

_suite = AnalysisSuite()


def basin_populations(
    x: Tensor,
    region_fns: dict[str, RegionFn],
    *,
    log_w: Tensor | None = None,
) -> dict[str, float]:
    """Compute basin populations from point assignments or importance weights."""
    return _suite.basin_populations(x, region_fns, log_w=log_w)


def rectangular_region(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> RegionFn:
    """Create a rectangular region predicate over 2D coordinates."""
    return AnalysisSuite.rectangular_region(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
