from __future__ import annotations
from collections.abc import Callable
import math
from typing import Any
from typing import cast
from typing import NamedTuple

import numpy as np

from optuna.distributions import CategoricalDistribution
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._utils import _get_skipped_trial_numbers
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical
from optuna.visualization._utils import _is_reverse_scale


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._utils import COLOR_SCALE

_logger = get_logger(__name__)


class _DimensionInfo(NamedTuple):
    label: str
    values: tuple[float, ...]
    range: tuple[float, float]
    is_log: bool
    is_cat: bool
    tickvals: list[int | float]
    ticktext: list[str]


class _ParallelCoordinateInfo(NamedTuple):
    dim_objective: _DimensionInfo
    dims_params: list[_DimensionInfo]
    reverse_scale: bool
    target_name: str


def plot_parallel_coordinate(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the high-dimensional parameter relationships in a study.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        params:
            Parameter list to visualize. The default is all parameters.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label and the legend.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.

    .. note::
        The colormap is reversed when the ``target`` argument isn't :obj:`None` or ``direction``
        of :class:`~optuna.study.Study` is ``minimize``.
    """

    _imports.check()
    info = _get_parallel_coordinate_info(study, params, target, target_name)
    return _get_parallel_coordinate_plot(info)


def _get_parallel_coordinate_plot(info: _ParallelCoordinateInfo) -> "go.Figure":
    layout = go.Layout(title="Parallel Coordinate Plot")

    if len(info.dims_params) == 0 or len(info.dim_objective.values) == 0:
        return go.Figure(data=[], layout=layout)

    dims = _get_dims_from_info(info)
    reverse_scale = info.reverse_scale
    target_name = info.target_name

    traces = [
        go.Parcoords(
            dimensions=dims,
            labelangle=30,
            labelside="bottom",
            line={
                "color": dims[0]["values"],
                "colorscale": COLOR_SCALE,
                "colorbar": {"title": target_name},
                "showscale": True,
                "reversescale": reverse_scale,
            },
        )
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure


def _get_parallel_coordinate_info(
    study: Study,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> _ParallelCoordinateInfo:
    _check_plot_args(study, target, target_name)

    reverse_scale = _is_reverse_scale(study, target)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    all_params = {p_name for t in trials for p_name in t.params}
    if params is not None:
        params_set = set(params)
        missing_params = [
            input_p_name for input_p_name in params if input_p_name not in all_params
        ]
        if missing_params:
            raise ValueError(
                "Parameter {} does not exist in your study.".format(missing_params[0])
            )
        all_params = params_set
    sorted_params = sorted(all_params)

    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    skipped_trial_numbers = _get_skipped_trial_numbers(trials, sorted_params)

    # Use generator expression directly for objectives
    objectives = tuple(target(t) for t in trials if t.number not in skipped_trial_numbers)
    objective_range = (min(objectives), max(objectives)) if objectives else (0, 0)
    dim_objective = _DimensionInfo(
        label=target_name,
        values=objectives,
        range=objective_range,
        is_log=False,
        is_cat=False,
        tickvals=[],
        ticktext=[],
    )

    if not trials:
        _logger.warning("Your study does not have any completed trials.")
        return _ParallelCoordinateInfo(
            dim_objective=dim_objective,
            dims_params=[],
            reverse_scale=reverse_scale,
            target_name=target_name,
        )

    if not objectives:
        _logger.warning("Your study has only completed trials with missing parameters.")
        return _ParallelCoordinateInfo(
            dim_objective=dim_objective,
            dims_params=[],
            reverse_scale=reverse_scale,
            target_name=target_name,
        )

    numeric_cat_params_indices: list[int] = []
    dims = []
    # Precompute set of skipped trial numbers for fast contains check
    skipped_set = skipped_trial_numbers
    for dim_index, p_name in enumerate(sorted_params, start=1):
        # Preallocate needed lists and variables
        is_categorical = False
        # Collect values for present trials only, skipping skipped trials in a single list comprehension
        values_and_types = [
            (t.params[p_name], isinstance(t.distributions[p_name], CategoricalDistribution))
            for t in trials
            if t.number not in skipped_set and p_name in t.params
        ]
        if values_and_types:
            values, types = zip(*values_and_types)
            is_categorical = any(types)
        else:
            values = ()
        if _is_log_scale(trials, p_name):
            log_values = [math.log10(v) for v in values]
            min_value = min(log_values)
            max_value = max(log_values)
            tickvals = list(range(math.ceil(min_value), math.floor(max_value) + 1))
            if min_value not in tickvals:
                tickvals = [min_value] + tickvals
            if max_value not in tickvals:
                tickvals = tickvals + [max_value]
            dim = _DimensionInfo(
                label=_truncate_label(p_name),
                values=tuple(log_values),
                range=(min_value, max_value),
                is_log=True,
                is_cat=False,
                tickvals=tickvals,
                ticktext=["{:.3g}".format(math.pow(10, x)) for x in tickvals],
            )
        elif is_categorical:
            vocab: dict[int | str, int] = {}
            # Faster one-pass build for numeric categoricals
            if _is_numerical(trials, p_name):
                sorted_unique = sorted(set(values))  # set: faster for unique values
                for idx, v in enumerate(sorted_unique):
                    vocab[v] = idx
                mapped_values = [vocab[v] for v in values]
                ticktext = [str(v) for v in sorted_unique]
                numeric_cat_params_indices.append(dim_index)
            else:
                # Map based on encounter order, keep ticktext in map order
                for v in values:
                    if v not in vocab:
                        vocab[v] = len(vocab)
                mapped_values = [vocab[v] for v in values]
                ticktext = [str(v) for v in sorted(vocab.keys(), key=lambda x: vocab[x])]
            dim = _DimensionInfo(
                label=_truncate_label(p_name),
                values=tuple(mapped_values),
                range=(min(mapped_values), max(mapped_values)) if mapped_values else (0, 0),
                is_log=False,
                is_cat=True,
                tickvals=list(range(len(vocab))),
                ticktext=ticktext,
            )
        else:
            float_values = tuple(values)
            mm = (min(float_values), max(float_values)) if float_values else (0, 0)
            dim = _DimensionInfo(
                label=_truncate_label(p_name),
                values=float_values,
                range=mm,
                is_log=False,
                is_cat=False,
                tickvals=[],
                ticktext=[],
            )

        dims.append(dim)

    if numeric_cat_params_indices:
        dims.insert(0, dim_objective)
        # np.lexsort consumes the sort keys the order from back to front.
        # So the values of parameters have to be reversed the order.
        idx = np.lexsort([dims[index].values for index in numeric_cat_params_indices][::-1])
        # Use np.take instead of manual array conversion
        updated_dims = [
            _DimensionInfo(
                label=dim.label,
                values=tuple(np.take(dim.values, idx)),
                range=dim.range,
                is_log=dim.is_log,
                is_cat=dim.is_cat,
                tickvals=dim.tickvals,
                ticktext=dim.ticktext,
            )
            for dim in dims
        ]
        dim_objective = updated_dims[0]
        dims = updated_dims[1:]

    return _ParallelCoordinateInfo(
        dim_objective=dim_objective,
        dims_params=dims,
        reverse_scale=reverse_scale,
        target_name=target_name,
    )


def _get_dims_from_info(info: _ParallelCoordinateInfo) -> list[dict[str, Any]]:
    dims = [
        {
            "label": info.dim_objective.label,
            "values": info.dim_objective.values,
            "range": info.dim_objective.range,
        }
    ]

    for dim in info.dims_params:
        if dim.is_log or dim.is_cat:
            dims.append(
                {
                    "label": dim.label,
                    "values": dim.values,
                    "range": dim.range,
                    "tickvals": dim.tickvals,
                    "ticktext": dim.ticktext,
                }
            )
        else:
            dims.append({"label": dim.label, "values": dim.values, "range": dim.range})

    return dims


def _truncate_label(label: str) -> str:
    return label if len(label) < 20 else "{}...".format(label[:17])
