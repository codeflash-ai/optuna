from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.0.0")
class SPXCrossover(BaseCrossover):
    """Simplex Crossover operation used by :class:`~optuna.samplers.NSGAIISampler`.

    Uniformly samples child individuals from within a single simplex
    that is similar to the simplex produced by the parent individual.
    For further information about SPX crossover, please refer to the following paper:

    - `Shigeyoshi Tsutsui and Shigeyoshi Tsutsui and David E. Goldberg and
      David E. Goldberg and Kumara Sastry and Kumara Sastry
      Progress Toward Linkage Learning in Real-Coded GAs with Simplex Crossover.
      IlliGAL Report. 2000.
      <https://www.researchgate.net/publication/2388486_Progress_Toward_Linkage_Learning_in_Real-Coded_GAs_with_Simplex_Crossover>`__

    Args:
        epsilon:
            Expansion rate. If not specified, defaults to ``sqrt(len(search_space) + 2)``.
    """

    n_parents = 3

    def __init__(self, epsilon: float | None = None) -> None:
        self._epsilon = epsilon

    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space_bounds: np.ndarray,
    ) -> np.ndarray:
        # https://www.researchgate.net/publication/2388486_Progress_Toward_Linkage_Learning_in_Real-Coded_GAs_with_Simplex_Crossover
        # Section 2 A Brief Review of SPX

        n = self.n_parents - 1
        G = np.mean(parents_params, axis=0, dtype=parents_params.dtype)  # Equation (1).
        rs = np.power(rng.rand(n), 1 / (np.arange(n) + 1))  # Equation (2).

        epsilon = np.sqrt(len(search_space_bounds) + 2) if self._epsilon is None else self._epsilon

        # Use numpy vectorization to compute all xks at once for better performance.
        # parents_params.shape = (n_parents, param_dim)
        # G.shape = (param_dim,)
        xks = G + epsilon * (parents_params - G)  # Shape: (n_parents, param_dim)

        # Vectorize calculation of ck to avoid Python loop
        # We're computing: ck = rs[0]*(xks[0]-xks[1]) + rs[1]*(xks[1]-xks[2] + ...) for three parents (n_parents=3)
        # Generalizing for any n_parents

        # xks shape: (n_parents, param_dim)
        # For k in 1..n_parents-1 (i.e. 1..n)
        # Precompute the deltas (xks[0]-xks[1], xks[1]-xks[2], ...)
        if self.n_parents == 3:
            # Common case: hardcoded for n_parents == 3 for best possible performance
            # Unroll the loop for n=2 (so k=1,2):
            # ck = rs[0] * (xks[0] - xks[1])
            # ck = rs[1] * (xks[1] - xks[2] + ck)
            # ==> ck = rs[1] * (xks[1] - xks[2] + rs[0] * (xks[0] - xks[1]))
            # Simplifies to:
            ck = rs[1] * (xks[1] - xks[2] + rs[0] * (xks[0] - xks[1]))
        else:
            # General case: use for-loop, but use in-place addition and limit temporary allocations
            ck = np.zeros_like(G)
            for k in range(1, self.n_parents):
                # Compute delta = xks[k-1] - xks[k]
                np.subtract(xks[k - 1], xks[k], out=ck)
                # Add previous ck
                if k > 1:
                    ck += prev_ck
                # Scale with rs[k-1]
                prev_ck = rs[k - 1] * ck
            ck = prev_ck  # type: ignore

        child_params = xks[-1] + ck  # Equation (5).

        return child_params
