"""Analysis utilities for weighted and unweighted population estimates."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from .sampling import SamplingEngine

RegionFn = Callable[[Tensor], Tensor]


class AnalysisSuite:
    """Basin populations, region predicates, and free-energy statistics."""

    def __init__(self, sampling_engine: SamplingEngine | None = None) -> None:
        self.sampling_engine = sampling_engine

    def basin_populations(
        self,
        x: Tensor,
        region_fns: dict[str, RegionFn],
        *,
        log_w: Tensor | None = None,
    ) -> dict[str, float]:
        """Compute basin populations from point assignments or importance weights."""
        if log_w is None:
            total = float(x.shape[0])
            return {
                name: float(region_fn(x).float().sum().item() / total)
                for name, region_fn in region_fns.items()
            }

        w = SamplingEngine.normalized_weights(log_w)
        out: dict[str, float] = {}
        for name, region_fn in region_fns.items():
            mask = region_fn(x).bool()
            out[name] = float(w[mask].sum().item())
        return out

    @staticmethod
    def rectangular_region(
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> RegionFn:
        """Create a rectangular region predicate over 2D coordinates."""

        def _region(x: Tensor) -> Tensor:
            return (x[:, 0] >= x_min) & (x[:, 0] < x_max) & (x[:, 1] >= y_min) & (x[:, 1] < y_max)

        return _region

    def free_energy_diff(
        self,
        x: Tensor,
        log_w: Tensor,
        region_a: RegionFn,
        region_b: RegionFn,
    ) -> float:
        """ΔF between two regions using importance weights."""
        return SamplingEngine.free_energy_diff(x, log_w, region_a, region_b)
