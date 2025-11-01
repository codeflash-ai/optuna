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
        # --- Optimized np.mean using axis sum and division for speed, as n_parents is small and likely 3
        G = np.sum(parents_params, axis=0) / self.n_parents  # Equation (1).

        # --- Precompute the exponent array outside np.power for speed
        exp_arr = 1 / (np.arange(n) + 1)
        rand_vals = rng.rand(n)
        rs = rand_vals**exp_arr  # Equation (2).

        epsilon = np.sqrt(len(search_space_bounds) + 2) if self._epsilon is None else self._epsilon

        # --- Vectorized xks calculation for speed and reduced intermediate allocations
        # parents_params shape: (n_parents, n_dimensions)
        # G shape: (n_dimensions,)
        # (pk - G): shape (n_parents, n_dimensions)
        # Broadcasting epsilon is fine, scalar.
        xks = G + epsilon * (parents_params - G)  # Equation (3).

        # --- Use array instead of loop for accumulation for better memory & efficiency

        ck = 0  # Equation (4).
        for k in range(1, self.n_parents):
            ck = rs[k - 1] * (xks[k - 1] - xks[k] + ck)

        child_params = xks[-1] + ck  # Equation (5).

        return child_params
