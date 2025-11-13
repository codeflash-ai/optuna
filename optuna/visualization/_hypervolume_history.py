from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from optuna._experimental import experimental_func
from optuna._hypervolume import compute_hypervolume
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _HypervolumeHistoryInfo(NamedTuple):
    trial_numbers: list[int]
    values: list[float]


@experimental_func("3.3.0")
def plot_hypervolume_history(
    study: Study,
    reference_point: Sequence[float],
) -> "go.Figure":
    """Plot hypervolume history of all trials in a study.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their hypervolumes.
            The number of objectives must be 2 or more.

        reference_point:
            A reference point to use for hypervolume computation.
            The dimension of the reference point must be the same as the number of objectives.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()

    if not study._is_multi_objective():
        raise ValueError(
            "Study must be multi-objective. For single-objective optimization, "
            "please use plot_optimization_history instead."
        )

    if len(reference_point) != len(study.directions):
        raise ValueError(
            "The dimension of the reference point must be the same as the number of objectives."
        )

    info = _get_hypervolume_history_info(study, np.asarray(reference_point, dtype=np.float64))
    return _get_hypervolume_history_plot(info)


def _get_hypervolume_history_plot(
    info: _HypervolumeHistoryInfo,
) -> "go.Figure":
    layout = go.Layout(
        title="Hypervolume History Plot",
        xaxis={"title": "Trial"},
        yaxis={"title": "Hypervolume"},
    )

    data = go.Scatter(
        x=info.trial_numbers,
        y=info.values,
        mode="lines+markers",
    )
    return go.Figure(data=data, layout=layout)


def _get_hypervolume_history_info(
    study: Study,
    reference_point: np.ndarray,
) -> _HypervolumeHistoryInfo:
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    if len(completed_trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    # Our hypervolume computation module assumes that all objectives are minimized.
    # Here we transform the objective values and the reference point.
    signs = np.asarray([1 if d == StudyDirection.MINIMIZE else -1 for d in study.directions])
    minimization_reference_point = signs * reference_point

    # Only feasible trials are considered in hypervolume computation.
    trial_numbers = []
    hypervolume_values = []
    best_trials_values_normalized: list[np.ndarray] = []
    hypervolume = 0.0
    best_trials_values_normalized_arr: np.ndarray | None = None

    for trial in completed_trials:
        trial_numbers.append(trial.number)

        has_constraints = _CONSTRAINTS_KEY in trial.system_attrs
        if has_constraints:
            constraints_values = trial.system_attrs[_CONSTRAINTS_KEY]
            if any(x > 0.0 for x in constraints_values):
                # The trial is infeasible.
                hypervolume_values.append(hypervolume)
                continue

        values_normalized = (signs * trial.values)[np.newaxis, :]
        if best_trials_values_normalized_arr is not None:
            # Best trials are stored as numpy array, fast computation with new trial
            # The trial is not on the Pareto front.
            if (best_trials_values_normalized_arr <= values_normalized).all(axis=1).any():
                # The trial is not on the Pareto front.
                hypervolume_values.append(hypervolume)
                continue

        if (values_normalized > minimization_reference_point).any():
            hypervolume_values.append(hypervolume)
            continue

        prod_v = np.prod(minimization_reference_point - values_normalized)
        hypervolume += prod_v
        if best_trials_values_normalized_arr is None:
            # Only first trial
            best_trials_values_normalized.append(values_normalized)
            best_trials_values_normalized_arr = np.vstack(best_trials_values_normalized)
        else:
            # Use list for interim construction, convert to array only when needed
            best_trials_values_normalized.append(values_normalized)
            best_trials_values_normalized_arr = np.vstack(best_trials_values_normalized)
            limited_sols = np.maximum(best_trials_values_normalized_arr[:-1], values_normalized)
            hypervolume -= compute_hypervolume(limited_sols, minimization_reference_point)
            # Find all Pareto-optimal survivors, keep those and append new one
            is_kept = (best_trials_values_normalized_arr[:-1] < values_normalized).any(axis=1)
            # Prepare new survivor list: survivors + current trial
            survivors = []
            for idx, keep in enumerate(is_kept):
                if keep:
                    survivors.append(best_trials_values_normalized_arr[:-1][idx])
            survivors.append(values_normalized)
            best_trials_values_normalized = survivors
            best_trials_values_normalized_arr = np.vstack(best_trials_values_normalized)
        hypervolume_values.append(hypervolume)

    if not best_trials_values_normalized:
        _logger.warning("Your study does not have any feasible trials.")

    return _HypervolumeHistoryInfo(trial_numbers, hypervolume_values)
