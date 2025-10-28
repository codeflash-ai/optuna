from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    from optuna.study import Study


_NUMERICAL_DISTRIBUTIONS = (
    FloatDistribution,
    IntDistribution,
)


def _try_crossover(
    parents: list[FrozenTrial],
    crossover: BaseCrossover,
    study: Study,
    rng: np.random.RandomState,
    swapping_prob: float,
    categorical_search_space: dict[str, BaseDistribution],
    numerical_search_space: dict[str, BaseDistribution],
    numerical_transform: _SearchSpaceTransform | None,
) -> dict[str, Any]:
    child_params: dict[str, Any] = {}

    if len(categorical_search_space) > 0:
        categorical_keys = list(categorical_search_space)
        # vectorized extraction of params for both parents
        parents_categorical_params = np.empty((2, len(categorical_keys)), dtype=object)
        params0 = parents[0].params
        params1 = parents[-1].params
        for i, k in enumerate(categorical_keys):
            parents_categorical_params[0, i] = params0[k]
            parents_categorical_params[1, i] = params1[k]

        child_categorical_array = _inlined_categorical_uniform_crossover(
            parents_categorical_params, rng, swapping_prob, categorical_search_space
        )
        child_categorical_params = dict(zip(categorical_keys, child_categorical_array))
        child_params.update(child_categorical_params)

    if numerical_transform is None:
        return child_params

    num_keys = list(numerical_search_space.keys())
    # Instead of running transform for all parents over all keys,
    # precompute the parameter lists:
    numerator_parents_params = np.empty((len(parents), len(num_keys)), dtype=object)
    for i, parent in enumerate(parents):
        parent_params = parent.params
        for j, key in enumerate(num_keys):
            numerator_parents_params[i, j] = parent_params[key]
    # Use a list-comprehension to avoid repeated dict construction overhead
    parents_numerical_params = np.stack(
        [
            numerical_transform.transform(
                {key: numerator_parents_params[i, j] for j, key in enumerate(num_keys)}
            )
            for i in range(len(parents))
        ]
    )
    child_numerical_array = crossover.crossover(
        parents_numerical_params, rng, study, numerical_transform.bounds
    )
    child_numerical_params = numerical_transform.untransform(child_numerical_array)
    child_params.update(child_numerical_params)

    return child_params


def perform_crossover(
    crossover: BaseCrossover,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    search_space: dict[str, BaseDistribution],
    rng: np.random.RandomState,
    swapping_prob: float,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> dict[str, Any]:
    # Materialize search_space.keys() and items once
    search_space_items = list(search_space.items())
    _numerical_types = _NUMERICAL_DISTRIBUTIONS
    numerical_search_space: dict[str, BaseDistribution] = {}
    categorical_search_space: dict[str, BaseDistribution] = {}
    for key, value in search_space_items:
        if isinstance(value, _numerical_types):
            numerical_search_space[key] = value
        else:
            categorical_search_space[key] = value

    numerical_transform: _SearchSpaceTransform | None = None
    if numerical_search_space:
        numerical_transform = _SearchSpaceTransform(numerical_search_space)

    # The while loop is an optimization hotspot.
    # For parent selection, we optimize the input set difference, see below.
    parents_indices_buffer = np.arange(len(parent_population))

    while True:
        parents = _select_parents_fast(
            crossover, study, parent_population, rng, dominates, parents_indices_buffer
        )
        child_params = _try_crossover(
            parents,
            crossover,
            study,
            rng,
            swapping_prob,
            categorical_search_space,
            numerical_search_space,
            numerical_transform,
        )
        if _is_contained_fast(child_params, search_space):
            break

    return child_params


def _select_parents(
    crossover: BaseCrossover,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> list[FrozenTrial]:
    parents: list[FrozenTrial] = []
    for _ in range(crossover.n_parents):
        parent = _select_parent(
            study, [t for t in parent_population if t not in parents], rng, dominates
        )
        parents.append(parent)

    return parents


def _select_parent(
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> FrozenTrial:
    population_size = len(parent_population)
    candidate0 = parent_population[rng.choice(population_size)]
    candidate1 = parent_population[rng.choice(population_size)]

    # TODO(ohta): Consider crowding distance.
    if dominates(candidate0, candidate1, study.directions):
        return candidate0
    else:
        return candidate1


def _is_contained(params: dict[str, Any], search_space: dict[str, BaseDistribution]) -> bool:
    for param_name in params.keys():
        param, param_distribution = params[param_name], search_space[param_name]

        if not param_distribution._contains(param_distribution.to_internal_repr(param)):
            return False
    return True


def _inlined_categorical_uniform_crossover(
    parent_params: np.ndarray,
    rng: np.random.RandomState,
    swapping_prob: float,
    search_space: dict[str, BaseDistribution],
) -> np.ndarray:
    # We can't use uniform crossover implementation of `BaseCrossover` for
    # parameters from `CategoricalDistribution`, since categorical params are
    # passed to crossover untransformed, which is not what `BaseCrossover`
    # implementations expect.
    n_categorical_params = len(search_space)
    masks = (rng.rand(n_categorical_params) >= swapping_prob).astype(int)
    return parent_params[masks, range(n_categorical_params)]


def _select_parents_fast(
    crossover: BaseCrossover,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
    parents_indices_buffer: np.ndarray,
) -> list[FrozenTrial]:
    # Avoid O(n^2) [t for t in parent_population if t not in parents]
    # by sampling indices without replacement
    n = len(parent_population)
    n_parents = crossover.n_parents
    # Use np.random.RandomState.choice for unique indices if possible
    # If n_parents == 2 (common), then we can optimize further
    if n_parents <= n:
        chosen_indices = rng.choice(n, size=n_parents, replace=False)
        return [parent_population[i] for i in chosen_indices]
    # fallback to original, slow logic
    parents: list[FrozenTrial] = []
    for _ in range(n_parents):
        parent = _select_parent(
            study, [t for t in parent_population if t not in parents], rng, dominates
        )
        parents.append(parent)
    return parents


def _is_contained_fast(params: dict[str, Any], search_space: dict[str, BaseDistribution]) -> bool:
    # Avoid .keys() and __getitem__ lookup costs
    for param_name, param in params.items():
        param_distribution = search_space[param_name]
        if not param_distribution._contains(param_distribution.to_internal_repr(param)):
            return False
    return True
