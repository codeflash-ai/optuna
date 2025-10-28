from __future__ import annotations

import numpy as np

from optuna.study._multi_objective import _is_pareto_front


def _compute_2d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    assert sorted_pareto_sols.shape[1] == reference_point.shape[0] == 2
    rect_diag_y = np.concatenate([reference_point[1:], sorted_pareto_sols[:-1, 1]])
    edge_length_x = reference_point[0] - sorted_pareto_sols[:, 0]
    edge_length_y = rect_diag_y - sorted_pareto_sols[:, 1]
    return edge_length_x @ edge_length_y


def _compute_3d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume in 3D. Time complexity is O(N^2) where N is sorted_pareto_sols.shape[0].
    If X, Y, Z coordinates are permutations of 0, 1, ..., N-1 and reference_point is (N, N, N), the
    hypervolume is calculated as the number of voxels (x, y, z) dominated by at least one point.
    If we fix x and y, this number is equal to the minimum of z' over all points (x', y', z')
    satisfying x' <= x and y' <= y. This can be efficiently computed using cumulative minimum
    (`np.minimum.accumulate`). Non-permutation coordinates can be transformed into permutation
    coordinates by using coordinate compression.
    """
    assert sorted_pareto_sols.shape[1] == reference_point.shape[0] == 3
    n = sorted_pareto_sols.shape[0]
    y_order = np.argsort(sorted_pareto_sols[:, 1])
    z_delta = np.zeros((n, n), dtype=float)
    z_delta[y_order, np.arange(n)] = reference_point[2] - sorted_pareto_sols[y_order, 2]
    z_delta = np.maximum.accumulate(np.maximum.accumulate(z_delta, axis=0), axis=1)
    # The x axis is already sorted, so no need to compress this coordinate.
    x_vals = sorted_pareto_sols[:, 0]
    y_vals = sorted_pareto_sols[y_order, 1]
    x_delta = np.concatenate([x_vals[1:], reference_point[:1]]) - x_vals
    y_delta = np.concatenate([y_vals[1:], reference_point[1:2]]) - y_vals
    # NOTE(nabenabe): Below is the faster alternative of `np.sum(dx[:, None] * dy * dz)`.
    return np.dot(np.dot(z_delta, y_delta), x_delta)


def _compute_hv(sorted_loss_vals: np.ndarray, reference_point: np.ndarray) -> float:
    if sorted_loss_vals.shape[0] == 1:
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        inclusive_hv = 1.0
        for r, v in zip(reference_point, sorted_loss_vals[0]):
            inclusive_hv *= r - v
        return float(inclusive_hv)
    elif sorted_loss_vals.shape[0] == 2:
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        # S(A v B) = S(A) + S(B) - S(A ^ B).
        hv1, hv2, intersec = 1.0, 1.0, 1.0
        for r, v1, v2 in zip(reference_point, sorted_loss_vals[0], sorted_loss_vals[1]):
            hv1 *= r - v1
            hv2 *= r - v2
            intersec *= r - max(v1, v2)
        return hv1 + hv2 - intersec

    inclusive_hvs = (reference_point - sorted_loss_vals).prod(axis=-1)
    # c.f. Eqs. (6) and (7) of ``A Fast Way of Calculating Exact Hypervolumes``.
    return inclusive_hvs[-1] + sum(
        _compute_exclusive_hv(
            np.maximum(sorted_loss_vals[i], sorted_loss_vals[i + 1 :]),
            inclusive_hvs[i],
            reference_point,
        )
        for i in range(inclusive_hvs.size - 1)
    )


def _compute_exclusive_hv(
    limited_sols: np.ndarray, inclusive_hv: float, reference_point: np.ndarray
) -> float:
    assert limited_sols.shape[0] >= 1
    n_points = limited_sols.shape[0]

    # Optimize case for <= 3 points: avoid Python for-loop in _compute_hv
    if n_points == 1:
        diff = reference_point - limited_sols[0]
        hv = float(np.prod(diff))
        return inclusive_hv - hv
    elif n_points == 2:
        v0, v1 = limited_sols[0], limited_sols[1]
        diff0 = reference_point - v0
        diff1 = reference_point - v1
        max_v0v1 = np.maximum(v0, v1)
        diff_intersec = reference_point - max_v0v1
        hv1 = np.prod(diff0)
        hv2 = np.prod(diff1)
        intersec = np.prod(diff_intersec)
        return inclusive_hv - (hv1 + hv2 - intersec)
    elif n_points == 3:
        # (slightly less preferred: reuse _compute_hv's optimized path for 3 points)
        # Vectorized inclusive_hvs
        inclusive_hvs = np.prod(reference_point - limited_sols, axis=-1)
        limited_sols_array = np.maximum(limited_sols[:, np.newaxis], limited_sols)
        total_hv = inclusive_hvs[-1]
        size = inclusive_hvs.size
        for i in range(size - 1):
            sli = limited_sols_array[i, i + 1 :]
            total_hv += _compute_exclusive_hv(sli, inclusive_hvs[i], reference_point)
        return inclusive_hv - total_hv

    # For general case, call Pareto filtering for maximal speed.
    on_front = _is_pareto_front(limited_sols, assume_unique_lexsorted=True)
    return inclusive_hv - _compute_hv(limited_sols[on_front], reference_point)


def compute_hypervolume(
    loss_vals: np.ndarray, reference_point: np.ndarray, assume_pareto: bool = False
) -> float:
    """Hypervolume calculator for any dimension.

    This class exactly calculates the hypervolume for any dimension.
    For 3 dimensions or higher, the WFG algorithm will be used.
    Please refer to ``A Fast Way of Calculating Exact Hypervolumes`` for the WFG algorithm.

    .. note::
        This class is used for computing the hypervolumes of points in multi-objective space.
        Each coordinate of each point represents a ``values`` of the multi-objective function.

    .. note::
        We check that each objective is to be minimized. Transform objective values that are
        to be maximized before calling this class's ``compute`` method.

    Args:
        loss_vals:
            An array of loss value vectors to calculate the hypervolume.
        reference_point:
            The reference point used to calculate the hypervolume.
        assume_pareto:
            Whether to assume the Pareto optimality to ``loss_vals``.
            In other words, if ``True``, none of loss vectors are dominated by another.
            ``assume_pareto`` is used only for speedup and it does not change the result even if
            this argument is wrongly given. If there are many non-Pareto solutions in
            ``loss_vals``, ``assume_pareto=True`` will speed up the calculation.

    Returns:
        The hypervolume of the given arguments.

    """

    if not np.all(loss_vals <= reference_point):
        raise ValueError(
            "All points must dominate or equal the reference point. "
            "That is, for all points in the loss_vals and the coordinate `i`, "
            "`loss_vals[i] <= reference_point[i]`."
        )
    if not np.all(np.isfinite(reference_point)):
        # reference_point does not have nan, thanks to the verification above.
        return float("inf")
    if loss_vals.size == 0:
        return 0.0

    if not assume_pareto:
        unique_lexsorted_loss_vals = np.unique(loss_vals, axis=0)
        on_front = _is_pareto_front(unique_lexsorted_loss_vals, assume_unique_lexsorted=True)
        sorted_pareto_sols = unique_lexsorted_loss_vals[on_front]
    else:
        # NOTE(nabenabe): The result of this function does not change both by
        # np.argsort(loss_vals[:, 0]) and np.unique(loss_vals, axis=0).
        # But many duplications in loss_vals significantly slows down the function.
        # TODO(nabenabe): Make an option to use np.unique.
        sorted_pareto_sols = loss_vals[loss_vals[:, 0].argsort()]

    if reference_point.shape[0] == 2:
        hv = _compute_2d(sorted_pareto_sols, reference_point)
    elif reference_point.shape[0] == 3:
        # NOTE: For 3D points, we always prefer _compute_3d to _compute_hv because the time
        # complexity of _compute_3d is O(N^2), while that of _compute_nd is \\Omega(N^3)
        # - It calls _compute_exclusive_hv with i points for i = 0, 1, ..., N-1
        # - _compute_exclusive_hv calls _is_pareto_front, which is quadratic
        #   with the number of points
        hv = _compute_3d(sorted_pareto_sols, reference_point)
    else:
        hv = _compute_hv(sorted_pareto_sols, reference_point)

    # NOTE(nabenabe): `nan` happens when inf - inf happens, but this is inf in hypervolume due to
    # the submodularity.
    return hv if np.isfinite(hv) else float("inf")
