from __future__ import annotations

from enum import IntEnum
import math
import threading
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntDistribution


if TYPE_CHECKING:
    import scipy.stats.qmc as qmc

    from optuna.trial import FrozenTrial
else:
    from optuna._imports import _LazyImport

    qmc = _LazyImport("scipy.stats.qmc")


_threading_lock = threading.Lock()


class _ScaleType(IntEnum):
    LINEAR = 0
    LOG = 1
    CATEGORICAL = 2


class SearchSpace:
    def __init__(
        self,
        optuna_search_space: dict[str, BaseDistribution],
    ) -> None:
        # Preallocate arrays based on search space length for efficiency
        n_params = len(optuna_search_space)
        self._optuna_search_space = optuna_search_space
        self._scale_types = np.empty(n_params, dtype=np.int64)
        self._bounds = np.empty((n_params, 2), dtype=float)
        self._steps = np.empty(n_params, dtype=float)
        items = list(optuna_search_space.values())

        # Loop optimization: avoid dict.values() in each iteration, use list
        for i in range(n_params):
            distribution = items[i]
            if isinstance(distribution, CategoricalDistribution):
                self._scale_types[i] = _ScaleType.CATEGORICAL
                self._bounds[i, 0] = 0.0
                self._bounds[i, 1] = len(distribution.choices)
                self._steps[i] = 1.0
            else:
                # Only two types (Float, Int) reach here
                is_log = getattr(distribution, "log", False)
                self._scale_types[i] = _ScaleType.LOG if is_log else _ScaleType.LINEAR
                self._bounds[i, 0] = distribution.low
                self._bounds[i, 1] = distribution.high
                step = getattr(distribution, "step", None)
                self._steps[i] = step if step is not None else 0.0
        self.dim = n_params
        # Avoid allocation for array comp: directly compare with constant
        # TODO: Make it an index array.
        self.is_categorical = self._scale_types == _ScaleType.CATEGORICAL
        # NOTE(nabenabe): MyPy Redefinition for NumPy v2.2.0. (Cast signed int to int)
        self.discrete_indices = np.flatnonzero(self._steps > 0).astype(int)
        self.continuous_indices = np.flatnonzero(self._steps == 0.0).astype(int)

    def get_normalized_params(
        self,
        trials: list[FrozenTrial],
    ) -> np.ndarray:
        values = np.empty((len(trials), len(self._optuna_search_space)), dtype=float)
        for i, (param, distribution) in enumerate(self._optuna_search_space.items()):
            if isinstance(distribution, CategoricalDistribution):
                values[:, i] = [distribution.to_internal_repr(t.params[param]) for t in trials]
            else:
                values[:, i] = _normalize_one_param(
                    np.array([trial.params[param] for trial in trials]),
                    self._scale_types[i],
                    (self._bounds[i, 0], self._bounds[i, 1]),
                    self._steps[i],
                )
        return values

    def get_unnormalized_param(
        self,
        normalized_param: np.ndarray,
    ) -> dict[str, Any]:
        # TODO(kAIto47802): Move the implementation of `_get_unnormalized_param` here
        # instead of wrapping it.
        return _get_unnormalized_param(self._optuna_search_space, normalized_param)

    def sample_normalized_params(self, n: int, rng: np.random.RandomState | None) -> np.ndarray:
        # TODO(kAIto47802): Move the implementation of `_sample_normalized_params` here
        # instead of wrapping it.
        return _sample_normalized_params(n, self, rng)

    def get_choices_of_discrete_params(self) -> list[np.ndarray]:
        return [
            (
                np.arange(self._bounds[i, 1])
                if self.is_categorical[i]
                else _normalize_one_param(
                    param_value=np.arange(
                        self._bounds[i, 0],
                        self._bounds[i, 1] + 0.5 * self._steps[i],
                        self._steps[i],
                    ),
                    scale_type=_ScaleType(self._scale_types[i]),
                    bounds=(self._bounds[i, 0], self._bounds[i, 1]),
                    step=self._steps[i],
                )
            )
            for i in self.discrete_indices
        ]


def _unnormalize_one_param(
    param_value: np.ndarray, scale_type: _ScaleType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    # param_value can be batched, or not.
    if scale_type == _ScaleType.CATEGORICAL:
        return param_value
    low, high = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
    if scale_type == _ScaleType.LOG:
        low, high = (math.log(low), math.log(high))
    param_value = param_value * (high - low) + low
    if scale_type == _ScaleType.LOG:
        param_value = np.exp(param_value)
    return param_value


def _normalize_one_param(
    param_value: np.ndarray, scale_type: _ScaleType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    # param_value can be batched, or not.
    if scale_type == _ScaleType.CATEGORICAL:
        return param_value
    low, high = (bounds[0] - 0.5 * step, bounds[1] + 0.5 * step)
    if scale_type == _ScaleType.LOG:
        low, high = (math.log(low), math.log(high))
        param_value = np.log(param_value)
    if high == low:
        return np.full_like(param_value, 0.5)
    param_value = (param_value - low) / (high - low)
    return param_value


def _round_one_normalized_param(
    param_value: np.ndarray, scale_type: _ScaleType, bounds: tuple[float, float], step: float
) -> np.ndarray:
    assert scale_type != _ScaleType.CATEGORICAL
    if step == 0.0:
        return param_value

    param_value = _unnormalize_one_param(param_value, scale_type, bounds, step)
    param_value = np.clip(
        (param_value - bounds[0] + 0.5 * step) // step * step + bounds[0],
        bounds[0],
        bounds[1],
    )
    param_value = _normalize_one_param(param_value, scale_type, bounds, step)
    return param_value


def _sample_normalized_params(
    n: int, search_space: SearchSpace, rng: np.random.RandomState | None
) -> np.ndarray:
    rng = rng or np.random.RandomState()
    dim = search_space._scale_types.shape[0]
    scale_types = search_space._scale_types
    bounds = search_space._bounds
    steps = search_space._steps

    # Sobol engine likely shares its internal state among threads.
    # Without threading.Lock, ValueError exceptions are raised in Sobol engine as discussed in
    # https://github.com/optuna/optunahub-registry/pull/168#pullrequestreview-2404054969
    with _threading_lock:
        qmc_engine = qmc.Sobol(dim, scramble=True, seed=rng.randint(np.iinfo(np.int32).max))
    param_values = qmc_engine.random(n)

    for i in range(dim):
        if scale_types[i] == _ScaleType.CATEGORICAL:
            param_values[:, i] = np.floor(param_values[:, i] * bounds[i, 1])
        elif steps[i] != 0.0:
            param_values[:, i] = _round_one_normalized_param(
                param_values[:, i], scale_types[i], (bounds[i, 0], bounds[i, 1]), steps[i]
            )
    return param_values


def _get_unnormalized_param(
    optuna_search_space: dict[str, BaseDistribution],
    normalized_param: np.ndarray,
) -> dict[str, Any]:
    ret = {}
    items = list(optuna_search_space.items())
    n_items = len(items)

    for i in range(n_items):
        param, distribution = items[i]
        if isinstance(distribution, CategoricalDistribution):
            ret[param] = distribution.to_external_repr(normalized_param[i])
        else:
            # FloatDistribution and IntDistribution only
            is_log = getattr(distribution, "log", False)
            scale_type = _ScaleType.LOG if is_log else _ScaleType.LINEAR
            step = getattr(distribution, "step", None)
            # Avoid constructing new tuple per iteration; can pass values directly
            low = distribution.low
            high = distribution.high
            step_value = 0.0 if step is None else step
            # Inline _unnormalize_one_param for single param case, reduces function call overhead
            value = normalized_param[i]
            if scale_type == _ScaleType.CATEGORICAL:
                param_value = value
            else:
                bound_low, bound_high = (low - 0.5 * step_value, high + 0.5 * step_value)
                if scale_type == _ScaleType.LOG:
                    bound_low, bound_high = np.log(bound_low), np.log(bound_high)
                param_value = value * (bound_high - bound_low) + bound_low
                if scale_type == _ScaleType.LOG:
                    param_value = np.exp(param_value)
                # np.clip as float for all
                param_value = float(np.clip(param_value, low, high))
            if isinstance(distribution, IntDistribution):
                param_value = round(param_value)
            ret[param] = param_value
    return ret
