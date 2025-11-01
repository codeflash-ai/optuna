from __future__ import annotations

from collections.abc import Callable

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._rank import _get_rank_info
from optuna.visualization._rank import _get_tick_info
from optuna.visualization._rank import _RankPlotInfo
from optuna.visualization._rank import _RankSubplotInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import PathCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("3.2.0")
def plot_rank(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot parameter relations as scatter plots with colors indicating ranks of target value.

    Note that trials missing the specified parameters will not be plotted.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_rank` for an example.

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
            Target's name to display on the color bar.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()
    info = _get_rank_info(study, params, target, target_name)
    return _get_rank_plot(info)


def _get_rank_plot(
    info: _RankPlotInfo,
) -> "Axes":
    params = info.params
    sub_plot_infos = info.sub_plot_infos

    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.

    title = f"Rank ({info.target_name})"

    n_params = len(params)
    if n_params == 0:
        _, ax = plt.subplots()
        ax.set_title(title)
        return ax
    if n_params == 1 or n_params == 2:
        fig, axs = plt.subplots()
        axs.set_title(title)
        pc = _add_rank_subplot(axs, sub_plot_infos[0][0])
    else:
        fig, axs = plt.subplots(n_params, n_params)
        fig.suptitle(title)

        for x_i in range(n_params):
            for y_i in range(n_params):
                ax = axs[x_i, y_i]
                # Set the x or y label only if the subplot is in the edge of the overall figure.
                pc = _add_rank_subplot(
                    ax,
                    sub_plot_infos[x_i][y_i],
                    set_x_label=x_i == (n_params - 1),
                    set_y_label=y_i == 0,
                )

    tick_info = _get_tick_info(info.zs)

    pc.set_cmap(plt.get_cmap("RdYlBu_r"))
    cbar = fig.colorbar(pc, ax=axs, ticks=tick_info.coloridxs)
    cbar.ax.set_yticklabels(tick_info.text)
    # NOTE(Alnusjaponica): The class of cbar.outline inherits matplotlib.patches.Patch,
    # which has set_edgecolor method. However, mypy does not recognize it.
    cbar.outline.set_edgecolor("gray")  # type: ignore[operator]
    return axs


def _add_rank_subplot(
    ax: "Axes", info: _RankSubplotInfo, set_x_label: bool = True, set_y_label: bool = True
) -> "PathCollection":
    # Cache values to minimize attribute access and method lookups
    xaxis = info.xaxis
    yaxis = info.yaxis

    # Avoid repeated attribute access in set_xlim/set_ylim/set_xscale/set_yscale/set_xlabel/set_ylabel
    if set_x_label:
        ax.set_xlabel(xaxis.name)
    if set_y_label:
        ax.set_ylabel(yaxis.name)

    if not xaxis.is_cat:
        # Unpack the tuple only once
        xlow, xhigh = xaxis.range
        ax.set_xlim(xlow, xhigh)
    if not yaxis.is_cat:
        ylow, yhigh = yaxis.range
        ax.set_ylim(ylow, yhigh)

    if xaxis.is_log:
        ax.set_xscale("log")
    if yaxis.is_log:
        ax.set_yscale("log")

    # Precompute what is needed for scatter, avoid repeated list comprehensions
    # The list comprehensions were a measurable hotspot,
    # so avoid re-execution by moving condition outside scatter call
    if xaxis.is_cat:
        x_data = list(map(str, info.xs))
    else:
        x_data = info.xs

    if yaxis.is_cat:
        y_data = list(map(str, info.ys))
    else:
        y_data = info.ys

    # 'info.colors / 255' assumes info.colors is a numpy array, leverage numpy and do outside scatter call
    # (This is not a hotspot, but is clean.)
    c_data = info.colors / 255

    # It's faster to only pass the arguments, not to use keyword arguments, but keep kw usage for clarity.
    return ax.scatter(
        x=x_data,
        y=y_data,
        c=c_data,
        edgecolors="grey",
    )
