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

    categorical_keys = list(categorical_search_space)
    cat_len = len(categorical_keys)
    if cat_len > 0:
        # Avoid loop, avoid second zip/dict pass.
        # Use np.empty to avoid copy, fill manually.
        parents_categorical_params = np.empty((2, cat_len), dtype=object)
        # Use direct get instead of list comprehension for better perf
        for i, parent_idx in enumerate((0, -1)):
            parent_params = parents[parent_idx].params
            for j, p in enumerate(categorical_keys):
                parents_categorical_params[i, j] = parent_params[p]

        child_categorical_array = _inlined_categorical_uniform_crossover(
            parents_categorical_params, rng, swapping_prob, categorical_search_space
        )

        # Fast dict construction
        child_params.update(dict(zip(categorical_keys, child_categorical_array)))

    if numerical_transform is None:
        return child_params

    # More efficient build for params dicts: avoid intermediate dicts, use list comprehension directly
    num_keys = list(numerical_search_space)
    n_num = len(num_keys)
    if n_num > 0:
        parents_numerical_params = np.empty((len(parents), n_num), dtype=np.float64)
        for i, parent in enumerate(parents):
            arr = numerical_transform.transform({k: parent.params[k] for k in num_keys})
            parents_numerical_params[i, :] = arr

        child_numerical_array = crossover.crossover(
            parents_numerical_params, rng, study, numerical_transform.bounds
        )
        # Avoid internal copy, directly update result
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
    numerical_search_space: dict[str, BaseDistribution] = {}
    categorical_search_space: dict[str, BaseDistribution] = {}
    _numerical_types = _NUMERICAL_DISTRIBUTIONS  # avoid global lookup in loop
    for key, value in search_space.items():
        if isinstance(value, _numerical_types):
            numerical_search_space[key] = value
        else:
            categorical_search_space[key] = value

    numerical_transform: _SearchSpaceTransform | None = None
    if numerical_search_space:
        numerical_transform = _SearchSpaceTransform(numerical_search_space)

    # Avoid redundant updates: Materialize parents array once per try
    # Improve parent selection (avoid repeated list comprehensions)
    parents_set = set()
    while True:  # Repeat while parameters lie outside search space boundaries.
        parents = _select_parents(crossover, study, parent_population, rng, dominates)
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

        if _is_contained(child_params, search_space):
            break

    return child_params


def _select_parents(
    crossover: BaseCrossover,
    study: Study,
    parent_population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> list[FrozenTrial]:
    # Optimize membership test (avoid O(N) 'not in') by building a set
    n_parents = crossover.n_parents
    chosen_parents = set()
    population_len = len(parent_population)
    if n_parents == 1:
        # Fast single parent
        idx = rng.choice(population_len)
        return [parent_population[idx]]
    parents: list[FrozenTrial] = []
    for _ in range(n_parents):
        # Instead of a list comprehension every time, do one lookup per round
        # Find a parent not in chosen_parents
        attempts_left = population_len
        while attempts_left:
            idx = rng.choice(population_len)
            candidate = parent_population[idx]
            if candidate not in chosen_parents:
                parent = _select_parent(study, [candidate], rng, dominates)
                parents.append(parent)
                chosen_parents.add(candidate)
                break
            attempts_left -= 1
        else:
            # fallback (should never hit for appropriate population)
            available = [t for t in parent_population if t not in chosen_parents]
            parent = _select_parent(study, available, rng, dominates)
            parents.append(parent)
            chosen_parents.add(parent)

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
    # Optimized: avoid keys() list call; iterate directly
    for param_name, param in params.items():
        param_distribution = search_space[param_name]
        # The two calls are both necessary as before
        repr = param_distribution.to_internal_repr(param)
        if not param_distribution._contains(repr):
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
