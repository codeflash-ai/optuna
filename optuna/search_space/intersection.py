from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import optuna
from optuna.distributions import BaseDistribution


if TYPE_CHECKING:
    from optuna.study import Study


def _calculate(
    trials: list[optuna.trial.FrozenTrial],
    include_pruned: bool = False,
    search_space: dict[str, BaseDistribution] | None = None,
    cached_trial_number: int = -1,
) -> tuple[dict[str, BaseDistribution] | None, int]:

    if include_pruned:
        states_of_interest = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.PRUNED,
        )
    else:
        states_of_interest = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.RUNNING,
        )

    next_cached_trial_number = -1

    for trial in reversed(trials):
        state = trial.state
        if state not in states_of_interest:
            continue

        if next_cached_trial_number == -1:
            next_cached_trial_number = trial.number + 1

        if cached_trial_number > trial.number:
            break

        if not state.is_finished():
            next_cached_trial_number = trial.number
            continue

        if search_space is None:
            # Only copy when search_space is first assigned:
            search_space = trial.distributions.copy()
            continue

        # Instead of filtering the whole dict every time, use set intersection for keys first:
        distributions = trial.distributions
        # Build a list of keys to keep
        keys_to_keep = []
        for name, distribution in search_space.items():
            try:
                if distributions[name] == distribution:
                    keys_to_keep.append(name)
            except KeyError:
                pass
        if len(keys_to_keep) == len(search_space):
            # No change, keep same object
            continue
        # Build new dict only if change detected
        search_space = {name: search_space[name] for name in keys_to_keep}

    return search_space, next_cached_trial_number


class IntersectionSearchSpace:
    """A class to calculate the intersection search space of a :class:`~optuna.study.Study`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Note that an instance of this class is supposed to be used for only one study.
    If different studies are passed to
    :func:`~optuna.search_space.IntersectionSearchSpace.calculate`,
    a :obj:`ValueError` is raised.

    Args:
        include_pruned:
            Whether pruned trials should be included in the search space.
    """

    def __init__(self, include_pruned: bool = False) -> None:
        self._cached_trial_number: int = -1
        self._search_space: dict[str, BaseDistribution] | None = None
        self._study_id: int | None = None

        self._include_pruned = include_pruned

    def calculate(self, study: Study, use_cache: bool = False) -> dict[str, BaseDistribution]:
        """Returns the intersection search space of the :class:`~optuna.study.Study`.

        Args:
            study:
                A study with completed trials. The same study must be passed for one instance
                of this class through its lifetime.
            use_cache:
                An option to use cached trials for each trial.

        Returns:
            A dictionary containing the parameter names and parameter's distributions sorted by
            parameter names.
        """

        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when
            # :class:`~optuna.storages.InMemoryStorage` is used because
            # :func:`~optuna.storages.InMemoryStorage.create_new_study`
            # always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`IntersectionSearchSpace` cannot handle multiple studies.")

        self._search_space, self._cached_trial_number = _calculate(
            study._get_trials(deepcopy=False, use_cache=use_cache),
            self._include_pruned,
            self._search_space,
            self._cached_trial_number,
        )
        search_space = self._search_space or {}
        search_space = dict(sorted(search_space.items(), key=lambda x: x[0]))
        return copy.deepcopy(search_space)


def intersection_search_space(
    trials: list[optuna.trial.FrozenTrial],
    include_pruned: bool = False,
) -> dict[str, BaseDistribution]:
    """Return the intersection search space of the given trials.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    .. note::
        :class:`~optuna.search_space.IntersectionSearchSpace` provides the same functionality with
        a much faster way. Please consider using it if you want to reduce execution time
        as much as possible.

    Args:
        trials:
            A list of trials.
        include_pruned:
            Whether pruned trials should be included in the search space.

    Returns:
        A dictionary containing the parameter names and parameter's distributions sorted by
        parameter names.
    """

    search_space, _ = _calculate(trials, include_pruned)
    search_space = search_space or {}
    # If already sorted or empty, return quickly
    if len(search_space) < 2:
        return search_space
    # Use an optimized sort for a dict with string keys
    items = search_space.items()
    search_space = dict(sorted(items, key=lambda x: x[0]))
    return search_space
