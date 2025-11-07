from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import numpy as np

from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._contour import _AxisInfo
from optuna.visualization._contour import _ContourInfo
from optuna.visualization._contour import _get_contour_info
from optuna.visualization._contour import _PlotValues
from optuna.visualization._contour import _SubContourInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports
import scipy


with try_import() as _optuna_imports:
    import scipy

if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import ContourSet
    from optuna.visualization.matplotlib._matplotlib_imports import plt


CONTOUR_POINT_NUM = 100


@experimental_func("2.2.0")
def plot_contour(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the parameter relationship as contour plot in a study with Matplotlib.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_contour` for an example.

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

    .. note::
        The colormap is reversed when the ``target`` argument isn't :obj:`None` or ``direction``
        of :class:`~optuna.study.Study` is ``minimize``.
    """

    _imports.check()
    info = _get_contour_info(study, params, target, target_name)
    return _get_contour_plot(info)


def _get_contour_plot(info: _ContourInfo) -> "Axes":
    sorted_params = info.sorted_params
    sub_plot_infos = info.sub_plot_infos
    reverse_scale = info.reverse_scale
    target_name = info.target_name

    if len(sorted_params) <= 1:
        _, ax = plt.subplots()
        return ax
    n_params = len(sorted_params)

    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    if n_params == 2:
        # Set up the graph style.
        fig, axs = plt.subplots()
        axs.set_title("Contour Plot")
        cmap = _set_cmap(reverse_scale)

        cs = _generate_contour_subplot(sub_plot_infos[0][0], axs, cmap)
        if isinstance(cs, ContourSet):
            axcb = fig.colorbar(cs)
            axcb.set_label(target_name)
    else:
        # Set up the graph style.
        fig, axs = plt.subplots(n_params, n_params)
        assert isinstance(axs, np.ndarray)
        fig.suptitle("Contour Plot")
        cmap = _set_cmap(reverse_scale)

        # Prepare data and draw contour plots.
        cs_list = []
        for x_i in range(len(sorted_params)):
            for y_i in range(len(sorted_params)):
                ax = axs[y_i, x_i]
                cs = _generate_contour_subplot(sub_plot_infos[y_i][x_i], ax, cmap)
                if isinstance(cs, ContourSet):
                    cs_list.append(cs)
        if cs_list:
            axcb = fig.colorbar(cs_list[0], ax=axs)
            axcb.set_label(target_name)

    return axs


def _set_cmap(reverse_scale: bool) -> "Colormap":
    cmap = "Blues_r" if not reverse_scale else "Blues"
    return plt.get_cmap(cmap)


class _LabelEncoder:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def fit(self, labels: list[str]) -> "_LabelEncoder":
        self.labels = sorted(set(labels))
        return self

    def transform(self, labels: list[str]) -> list[int]:
        return [self.labels.index(label) for label in labels]

    def fit_transform(self, labels: list[str]) -> list[int]:
        return self.fit(labels).transform(labels)

    def get_labels(self) -> list[str]:
        return self.labels

    def get_indices(self) -> list[int]:
        return list(range(len(self.labels)))


def _filter_missing_values(
    xaxis: _AxisInfo, yaxis: _AxisInfo
) -> tuple[list[str | float], list[str | float]]:
    x_values = []
    y_values = []
    for x_value, y_value in zip(xaxis.values, yaxis.values):
        if x_value is not None and y_value is not None:
            x_values.append(x_value)
            y_values.append(y_value)
    return x_values, y_values


def _calculate_axis_data(
    axis: _AxisInfo,
    values: Sequence[str | float],
) -> tuple[np.ndarray, list[str], list[int], list[int | float]]:
    # Convert categorical values to int.
    cat_param_labels: list[str] = []
    cat_param_pos: list[int] = []
    returned_values: Sequence[int | float]
    if axis.is_cat:
        enc = _LabelEncoder()
        # Fit LabelEncoder with all the categories in categorical distribution.
        all_axis_values = [str(value) for value in axis.values if value is not None]
        enc.fit(all_axis_values)
        # Then transform the values using the fitted label encoder.
        returned_values = enc.transform([str(value) for value in values])
        cat_param_labels = enc.get_labels()
        cat_param_pos = enc.get_indices()
    else:
        returned_values = [float(x) for x in values]

    # For x and y, create 1-D array of evenly spaced coordinates on linear or log scale.

    # For x and y, create 1-D array of evenly spaced coordinates on linear or log scale.
    if axis.is_log:
        ci = np.logspace(np.log10(axis.range[0]), np.log10(axis.range[1]), CONTOUR_POINT_NUM)
    else:
        ci = np.linspace(axis.range[0], axis.range[1], CONTOUR_POINT_NUM)

    return ci, cat_param_labels, cat_param_pos, list(returned_values)


def _calculate_griddata(info: _SubContourInfo) -> tuple[np.ndarray, _PlotValues, _PlotValues]:
    xaxis = info.xaxis
    yaxis = info.yaxis
    z_values_dict = info.z_values

    # Precompute non-None indices, and use NumPy arrays for faster filtering and indexing
    xaxis_vals = np.array(xaxis.values)
    yaxis_vals = np.array(yaxis.values)
    # Find positions where both x and y are not None
    mask_valid = (xaxis_vals != None) & (yaxis_vals != None)
    # It is faster to use np.where, but since these values can be str/float, we'll use tolist()
    x_values = xaxis_vals[mask_valid].tolist()
    y_values = yaxis_vals[mask_valid].tolist()

    # If no valid values, return empty values
    if not x_values or not y_values:
        return np.array([]), _PlotValues([], []), _PlotValues([], [])

    # For indices lookup, precompute both as sets for O(1) lookup.
    xindex_lookup = {val: idx for idx, val in enumerate(xaxis.indices)}
    yindex_lookup = {val: idx for idx, val in enumerate(yaxis.indices)}
    # Use list comprehension rather than zip and append for faster creation.
    z_values = [
        z_values_dict[(xindex_lookup[x_value], yindex_lookup[y_value])]
        for x_value, y_value in zip(x_values, y_values)
    ]

    xi, cat_param_labels_x, cat_param_pos_x, transformed_x_values = _calculate_axis_data(
        xaxis,
        x_values,
    )
    yi, cat_param_labels_y, cat_param_pos_y, transformed_y_values = _calculate_axis_data(
        yaxis,
        y_values,
    )

    # Calculate grid data points.
    zi: np.ndarray = np.array([])
    # Create irregularly spaced map of trial values
    # and interpolate it with Plotly's interpolation formulation.
    if xaxis.name != yaxis.name:
        zmap = _create_zmap(transformed_x_values, transformed_y_values, z_values, xi, yi)
        zi = _interpolate_zmap(zmap, CONTOUR_POINT_NUM)

    # categorize by constraints
    constraints_arr = np.array(info.constraints)
    feasible_x = []
    feasible_y = []
    infeasible_x = []
    infeasible_y = []
    for x_value, y_value, c in zip(transformed_x_values, transformed_y_values, constraints_arr):
        if c:
            feasible_x.append(x_value)
            feasible_y.append(y_value)
        else:
            infeasible_x.append(x_value)
            infeasible_y.append(y_value)

    feasible = _PlotValues(feasible_x, feasible_y)
    infeasible = _PlotValues(infeasible_x, infeasible_y)

    return zi, feasible, infeasible


def _generate_contour_subplot(
    info: _SubContourInfo, ax: "Axes", cmap: "Colormap"
) -> "ContourSet" | None:
    ax.label_outer()

    if len(info.xaxis.indices) < 2 or len(info.yaxis.indices) < 2:
        return None

    ax.set(xlabel=info.xaxis.name, ylabel=info.yaxis.name)
    ax.set_xlim(info.xaxis.range[0], info.xaxis.range[1])
    ax.set_ylim(info.yaxis.range[0], info.yaxis.range[1])
    x_values, y_values = _filter_missing_values(info.xaxis, info.yaxis)
    xi, x_cat_param_label, x_cat_param_pos, _ = _calculate_axis_data(info.xaxis, x_values)
    yi, y_cat_param_label, y_cat_param_pos, _ = _calculate_axis_data(info.yaxis, y_values)
    if info.xaxis.is_cat:
        ax.set_xticks(x_cat_param_pos)
        ax.set_xticklabels(x_cat_param_label)
    else:
        ax.set_xscale("log" if info.xaxis.is_log else "linear")
    if info.yaxis.is_cat:
        ax.set_yticks(y_cat_param_pos)
        ax.set_yticklabels(y_cat_param_label)
    else:
        ax.set_yscale("log" if info.yaxis.is_log else "linear")

    if info.xaxis.name == info.yaxis.name:
        return None

    zi, feasible_plot_values, infeasible_plot_values = _calculate_griddata(info)
    cs = None
    if len(zi) > 0:
        # Contour the gridded data.
        ax.contour(xi, yi, zi, 15, linewidths=0.5, colors="k")
        cs = ax.contourf(xi, yi, zi, 15, cmap=cmap.reversed())
        assert isinstance(cs, ContourSet)
        # Plot data points.
        ax.scatter(
            feasible_plot_values.x,
            feasible_plot_values.y,
            marker="o",
            c="black",
            s=20,
            edgecolors="grey",
            linewidth=2.0,
        )
        ax.scatter(
            infeasible_plot_values.x,
            infeasible_plot_values.y,
            marker="o",
            c="#cccccc",
            s=20,
            edgecolors="grey",
            linewidth=2.0,
        )

    return cs


def _create_zmap(
    x_values: Sequence[int | float],
    y_values: Sequence[int | float],
    z_values: Sequence[float],
    xi: np.ndarray,
    yi: np.ndarray,
) -> dict[tuple[int, int], float]:
    # Use NumPy arrays for candidate value computation.
    xi_arr = xi
    yi_arr = yi
    zmap = {}
    # Vectorized computation for index lookup (argmin over array)
    x_array = np.array(x_values)
    y_array = np.array(y_values)
    z_array = np.array(z_values)
    # Below, vectorize as much as possible
    x_indices = np.abs(xi_arr[np.newaxis, :] - x_array[:, np.newaxis])
    x_min = np.argmin(x_indices, axis=1)
    y_indices = np.abs(yi_arr[np.newaxis, :] - y_array[:, np.newaxis])
    y_min = np.argmin(y_indices, axis=1)
    # Dictionary from tuple index to z value. No duplicate keys in zipped loop.
    for idx in range(len(z_array)):
        zmap[(x_min[idx], y_min[idx])] = z_array[idx]

    return zmap


def _interpolate_zmap(zmap: dict[tuple[int, int], float], contour_plot_num: int) -> np.ndarray:
    # Implements interpolation formulation used in Plotly
    # to interpolate heatmaps and contour plots
    # https://github.com/plotly/plotly.js/blob/95b3bd1bb19d8dc226627442f8f66bce9576def8/src/traces/heatmap/interp2d.js#L15-L20
    # citing their doc:
    #
    # > Fill in missing data from a 2D array using an iterative
    # > poisson equation solver with zero-derivative BC at edges.
    # > Amazingly, this just amounts to repeatedly averaging all the existing
    # > nearest neighbors
    #
    # Plotly's algorithm is equivalent to solve the following linear simultaneous equation.
    # It is discretization form of the Poisson equation.
    #
    #     z[x, y] = zmap[(x, y)]                                  (if zmap[(x, y)] is given)
    # 4 * z[x, y] = z[x-1, y] + z[x+1, y] + z[x, y-1] + z[x, y+1] (if zmap[(x, y)] is not given)

    # Preallocate arrays (avoid repeated list appends)
    sz = contour_plot_num
    N = sz * sz
    b = np.zeros(N)
    # We know the matrix will be at most 5 nonzeroes per row (interior)
    # Use lists for COO format
    a_data = []
    a_row = []
    a_col = []
    # Vectorize neighbor offsets to avoid repeated for-loop lookups
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Precompute keys for fast lookup
    zmap_keys = set(zmap.keys())
    for x in range(sz):
        for y in range(sz):
            grid_index = y * sz + x
            if (x, y) in zmap_keys:
                a_data.append(1)
                a_row.append(grid_index)
                a_col.append(grid_index)
                b[grid_index] = zmap[(x, y)]
            else:
                a_data.append(4)
                a_row.append(grid_index)
                a_col.append(grid_index)
                for dx, dy in offsets:
                    xn = x + dx
                    yn = y + dy
                    if 0 <= xn < sz and 0 <= yn < sz:
                        neighbor_index = yn * sz + xn
                        a_data.append(-1)
                        a_row.append(grid_index)
                        a_col.append(neighbor_index)

    A = scipy.sparse.csc_matrix((a_data, (a_row, a_col)), shape=(N, N))
    z = scipy.sparse.linalg.spsolve(A, b)

    return z.reshape((sz, sz))
