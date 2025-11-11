from __future__ import annotations

import datetime
from typing import NamedTuple

from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _make_hovertext


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _TimelineBarInfo(NamedTuple):
    number: int
    start: datetime.datetime
    complete: datetime.datetime
    state: TrialState
    hovertext: str
    infeasible: bool


class _TimelineInfo(NamedTuple):
    bars: list[_TimelineBarInfo]


def plot_timeline(study: Study, n_recent_trials: int | None = None) -> "go.Figure":
    """Plot the timeline of a study.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted with
            their lifetime.
        n_recent_trials:
            The number of recent trials to plot. If :obj:`None`, all trials are plotted.
            If specified, only the most recent ``n_recent_trials`` will be displayed.
            Must be a positive integer.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.

    Raises:
        ValueError: if ``n_recent_trials`` is 0 or negative.
    """

    if n_recent_trials is not None and n_recent_trials <= 0:
        raise ValueError("n_recent_trials must be a positive integer or None.")

    _imports.check()
    info = _get_timeline_info(study, n_recent_trials=n_recent_trials)
    return _get_timeline_plot(info)


def _get_max_datetime_complete(study: Study) -> datetime.datetime:
    # Cache study.trials only once (it's a property, likely to be fast but best to ensure only one access).
    trials = study.trials

    # Combine loops to only iterate study.trials once and generate all needed data in a single traversal.
    max_run_duration: datetime.timedelta | None = None
    any_running_trials = False
    max_datetime_complete: datetime.datetime | None = None

    now = None  # Will only compute now() if needed.

    for t in trials:
        ds = t.datetime_start
        dc = t.datetime_complete

        if dc is not None and ds is not None:
            dur = dc - ds
            if (max_run_duration is None) or (dur > max_run_duration):
                max_run_duration = dur

        if dc is not None:
            if (max_datetime_complete is None) or (dc > max_datetime_complete):
                max_datetime_complete = dc

    # Optimize _is_running_trials_in_study logic in-place.
    running_trials = study.get_trials(states=(TrialState.RUNNING,), deepcopy=False)
    if max_run_duration is None:
        return (
            datetime.datetime.now()
            if running_trials
            else (
                datetime.datetime.now() if max_datetime_complete is None else max_datetime_complete
            )
        )

    if running_trials:
        if now is None:
            now = datetime.datetime.now()
        mult = 5 * max_run_duration
        for t in running_trials:
            ds = t.datetime_start
            if ds is not None and now - ds < mult:
                return datetime.datetime.now()

    # Return max datetime_complete, fallback to now
    return max_datetime_complete if max_datetime_complete is not None else datetime.datetime.now()


def _is_running_trials_in_study(study: Study, max_run_duration: datetime.timedelta | None) -> bool:
    running_trials = study.get_trials(states=(TrialState.RUNNING,), deepcopy=False)
    if max_run_duration is None:
        return len(running_trials) > 0

    now = datetime.datetime.now()
    # This heuristic is to check whether we have trials that were somehow killed,
    # still remain as `RUNNING` in `study`.
    return any(
        now - t.datetime_start < 5 * max_run_duration
        for t in running_trials
        # MyPy redefinition: Running trial should have datetime_start.
        if t.datetime_start is not None
    )


def _get_timeline_info(study: Study, n_recent_trials: int | None = None) -> _TimelineInfo:
    bars = []

    max_datetime = _get_max_datetime_complete(study)
    timedelta_for_small_bar = datetime.timedelta(seconds=1)

    trials = study.get_trials(deepcopy=False)
    if n_recent_trials is not None:
        trials = trials[-n_recent_trials:]

    for trial in trials:
        datetime_start = trial.datetime_start or max_datetime
        datetime_complete = (
            max_datetime + timedelta_for_small_bar
            if trial.state == TrialState.RUNNING
            else trial.datetime_complete or datetime_start + timedelta_for_small_bar
        )
        infeasible = (
            False
            if _CONSTRAINTS_KEY not in trial.system_attrs
            else any([x > 0 for x in trial.system_attrs[_CONSTRAINTS_KEY]])
        )
        if datetime_complete < datetime_start:
            _logger.warning(
                (
                    f"The start and end times for Trial {trial.number} seem to be reversed. "
                    f"The start time is {datetime_start} and the end time is {datetime_complete}."
                )
            )
        bars.append(
            _TimelineBarInfo(
                number=trial.number,
                start=datetime_start,
                complete=datetime_complete,
                state=trial.state,
                hovertext=_make_hovertext(trial),
                infeasible=infeasible,
            )
        )

    if len(bars) == 0:
        _logger.warning("Your study does not have any trials.")

    return _TimelineInfo(bars)


def _get_timeline_plot(info: _TimelineInfo) -> "go.Figure":
    _cm = {
        "COMPLETE": "blue",
        "FAIL": "red",
        "PRUNED": "orange",
        "RUNNING": "green",
        "WAITING": "gray",
    }

    fig = go.Figure()
    for state in sorted(TrialState, key=lambda x: x.name):
        if state.name == "COMPLETE":
            infeasible_bars = [b for b in info.bars if b.state == state and b.infeasible]
            feasible_bars = [b for b in info.bars if b.state == state and not b.infeasible]
            _plot_bars(infeasible_bars, "#cccccc", "INFEASIBLE", fig)
            _plot_bars(feasible_bars, _cm[state.name], state.name, fig)
        else:
            bars = [b for b in info.bars if b.state == state]
            _plot_bars(bars, _cm[state.name], state.name, fig)
    fig.update_xaxes(type="date")
    fig.update_layout(
        go.Layout(
            title="Timeline Plot",
            xaxis={"title": "Datetime"},
            yaxis={"title": "Trial"},
        )
    )
    fig.update_layout(showlegend=True)  # Draw a legend even if all TrialStates are the same.
    return fig


def _plot_bars(bars: list[_TimelineBarInfo], color: str, name: str, fig: go.Figure) -> None:
    if len(bars) == 0:
        return

    fig.add_trace(
        go.Bar(
            name=name,
            x=[(b.complete - b.start).total_seconds() * 1000 for b in bars],
            y=[b.number for b in bars],
            base=[b.start.isoformat() for b in bars],
            text=[b.hovertext for b in bars],
            hovertemplate="%{text}<extra>" + name + "</extra>",
            orientation="h",
            marker=dict(color=color),
            textposition="none",  # Avoid drawing hovertext in a bar.
        )
    )
