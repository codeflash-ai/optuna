from __future__ import annotations

import math
from typing import Any

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution


class _SearchSpaceTransform:
    """Transform a search space and parameter configurations to continuous space.

    The search space bounds and parameter configurations are represented as ``numpy.ndarray``s and
    transformed into continuous space. Bounds and parameters associated with categorical
    distributions are one-hot encoded. Parameter configurations in this space can additionally be
    untransformed, or mapped back to the original space. This type of
    transformation/untransformation is useful for e.g. implementing samplers without having to
    condition on distribution types before sampling parameter values.

    Args:
        search_space:
            The search space. If any transformations are to be applied, parameter configurations
            are assumed to hold parameter values for all of the distributions defined in this
            search space. Otherwise, assertion failures will be raised.
        transform_log:
            If :obj:`True`, apply log/exp operations to the bounds and parameters with
            corresponding distributions in log space during transformation/untransformation.
            Should always be :obj:`True` if any parameters are going to be sampled from the
            transformed space.
        transform_step:
            If :obj:`True`, offset the lower and higher bounds by a half step each, increasing the
            space by one step. This allows fair sampling for values close to the bounds.
            Should always be :obj:`True` if any parameters are going to be sampled from the
            transformed space.
        transform_0_1:
            If :obj:`True`, apply a linear transformation to the bounds and parameters so that
            they are in the unit cube.

    Attributes:
        bounds:
            Constructed bounds from the given search space.
        column_to_encoded_columns:
            Constructed mapping from original parameter column index to encoded column indices.
        encoded_column_to_column:
            Constructed mapping from encoded column index to original parameter column index.

    Note:
        Parameter values are not scaled to the unit cube.

    Note:
        ``transform_log`` and ``transform_step`` are useful for constructing bounds and parameters
        without any actual transformations by setting those arguments to :obj:`False`. This is
        needed for e.g. the hyperparameter importance assessments.

    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        transform_log: bool = True,
        transform_step: bool = True,
        transform_0_1: bool = False,
    ) -> None:
        bounds, column_to_encoded_columns, encoded_column_to_column = _transform_search_space(
            search_space, transform_log, transform_step
        )
        self._raw_bounds = bounds
        self._column_to_encoded_columns = column_to_encoded_columns
        self._encoded_column_to_column = encoded_column_to_column
        self._search_space = search_space
        self._transform_log = transform_log
        self._transform_0_1 = transform_0_1

    @property
    def bounds(self) -> np.ndarray:
        if self._transform_0_1:
            return np.array([[0.0, 1.0]] * self._raw_bounds.shape[0])
        else:
            return self._raw_bounds

    @property
    def column_to_encoded_columns(self) -> list[np.ndarray]:
        return self._column_to_encoded_columns

    @property
    def encoded_column_to_column(self) -> np.ndarray:
        return self._encoded_column_to_column

    def transform(self, params: dict[str, Any]) -> np.ndarray:
        """Transform a parameter configuration from actual values to continuous space.

        Args:
            params:
                A parameter configuration to transform.

        Returns:
            A 1-dimensional ``numpy.ndarray`` holding the transformed parameters in the
            configuration.

        """
        trans_params = np.zeros(self._raw_bounds.shape[0], dtype=np.float64)

        bound_idx = 0
        for name, distribution in self._search_space.items():
            assert name in params, "Parameter configuration must contain all distributions."
            param = params[name]

            if isinstance(distribution, CategoricalDistribution):
                choice_idx = int(distribution.to_internal_repr(param))
                trans_params[bound_idx + choice_idx] = 1
                bound_idx += len(distribution.choices)
            else:
                trans_params[bound_idx] = _transform_numerical_param(
                    param, distribution, self._transform_log
                )
                bound_idx += 1

        if self._transform_0_1:
            single_mask = self._raw_bounds[:, 0] == self._raw_bounds[:, 1]
            trans_params[single_mask] = 0.5
            trans_params[~single_mask] = (
                trans_params[~single_mask] - self._raw_bounds[~single_mask, 0]
            ) / (self._raw_bounds[~single_mask, 1] - self._raw_bounds[~single_mask, 0])

        return trans_params

    def untransform(self, trans_params: np.ndarray) -> dict[str, Any]:
        """Untransform a parameter configuration from continuous space to actual values.

        Args:
            trans_params:
                A 1-dimensional ``numpy.ndarray`` in the transformed space corresponding to a
                parameter configuration.

        Returns:
            A dictionary of an untransformed parameter configuration. Keys are parameter names.
            Values are untransformed parameter values.

        """
        assert trans_params.shape == (self._raw_bounds.shape[0],)

        if self._transform_0_1:
            trans_params = self._raw_bounds[:, 0] + trans_params * (
                self._raw_bounds[:, 1] - self._raw_bounds[:, 0]
            )

        params = {}

        for (name, distribution), encoded_columns in zip(
            self._search_space.items(), self.column_to_encoded_columns
        ):
            trans_param = trans_params[encoded_columns]

            if isinstance(distribution, CategoricalDistribution):
                # Select the highest rated one-hot encoding.
                param = distribution.to_external_repr(trans_param.argmax())
            else:
                param = _untransform_numerical_param(
                    trans_param.item(), distribution, self._transform_log
                )

            params[name] = param

        return params


def _transform_search_space(
    search_space: dict[str, BaseDistribution], transform_log: bool, transform_step: bool
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    assert len(search_space) > 0, "Cannot transform if no distributions are given."

    # Precompute Is-Categorical and n_choices for all distributions in one pass
    is_categorical = []
    n_bounds = 0
    choices_lens = []
    distributions = list(search_space.values())
    for d in distributions:
        cat = isinstance(d, CategoricalDistribution)
        is_categorical.append(cat)
        if cat:
            n = len(d.choices)
            n_bounds += n
            choices_lens.append(n)
        else:
            n_bounds += 1
            choices_lens.append(None)  # placeholder

    bounds = np.empty((n_bounds, 2), dtype=np.float64)
    column_to_encoded_columns: list[np.ndarray] = []
    encoded_column_to_column = np.empty(n_bounds, dtype=np.int64)

    bound_idx = 0
    idx = 0
    for d in distributions:
        if is_categorical[idx]:
            n_choices = choices_lens[idx]
            # Use slice assignment for bounds, np.arange for encoded_columns
            bounds[bound_idx : bound_idx + n_choices] = (0, 1)  # Broadcast across all choices.
            encoded_columns = np.arange(bound_idx, bound_idx + n_choices)
            encoded_column_to_column[encoded_columns] = len(column_to_encoded_columns)
            column_to_encoded_columns.append(encoded_columns)
            bound_idx += n_choices
        elif isinstance(d, FloatDistribution):
            # Remove redundant isinstance calls -- fold them into the distribution's type detection above
            if d.step is not None:
                half_step = 0.5 * d.step if transform_step else 0.0
                low = _transform_numerical_param(d.low, d, transform_log)
                high = _transform_numerical_param(d.high, d, transform_log)
                bds_low = low - half_step
                bds_high = high + half_step
            else:
                bds_low = _transform_numerical_param(d.low, d, transform_log)
                bds_high = _transform_numerical_param(d.high, d, transform_log)
            bounds[bound_idx] = (bds_low, bds_high)
            encoded_column = np.array([bound_idx], dtype=np.int64)
            encoded_column_to_column[bound_idx] = len(column_to_encoded_columns)
            column_to_encoded_columns.append(encoded_column)
            bound_idx += 1
        elif isinstance(d, IntDistribution):
            half_step = 0.5 * d.step if transform_step else 0.0
            if d.log:
                low = _transform_numerical_param(d.low - half_step, d, transform_log)
                high = _transform_numerical_param(d.high + half_step, d, transform_log)
                bds_low = low
                bds_high = high
            else:
                low = _transform_numerical_param(d.low, d, transform_log)
                high = _transform_numerical_param(d.high, d, transform_log)
                bds_low = low - half_step
                bds_high = high + half_step
            bounds[bound_idx] = (bds_low, bds_high)
            encoded_column = np.array([bound_idx], dtype=np.int64)
            encoded_column_to_column[bound_idx] = len(column_to_encoded_columns)
            column_to_encoded_columns.append(encoded_column)
            bound_idx += 1
        else:
            assert False, "Should not reach. Unexpected distribution."

        idx += 1

    assert bound_idx == n_bounds

    return bounds, column_to_encoded_columns, encoded_column_to_column


def _transform_numerical_param(
    param: int | float, distribution: BaseDistribution, transform_log: bool
) -> float:
    d = distribution

    # Avoid redundant isinstance checks, re-order them to minimize checks
    if isinstance(d, FloatDistribution) or isinstance(d, IntDistribution):
        # Check the log property directly
        if getattr(d, "log", False):
            return math.log(param) if transform_log else float(param)
        else:
            return float(param)
    elif isinstance(d, CategoricalDistribution):
        assert False, "Should not reach. Should be one-hot encoded."
    else:
        assert False, "Should not reach. Unexpected distribution."


def _untransform_numerical_param(
    trans_param: float, distribution: BaseDistribution, transform_log: bool
) -> int | float:
    d = distribution

    if isinstance(d, CategoricalDistribution):
        assert False, "Should not reach. Should be one-hot encoded."
    elif isinstance(d, FloatDistribution):
        if d.log:
            param = math.exp(trans_param) if transform_log else trans_param
            if d.single():
                pass
            else:
                param = min(param, np.nextafter(d.high, d.high - 1))
        elif d.step is not None:
            param = float(
                np.clip(np.round((trans_param - d.low) / d.step) * d.step + d.low, d.low, d.high)
            )
        else:
            if d.single():
                param = trans_param
            else:
                param = min(trans_param, np.nextafter(d.high, d.high - 1))
    elif isinstance(d, IntDistribution):
        if d.log:
            if transform_log:
                param = int(np.clip(np.round(math.exp(trans_param)), d.low, d.high))
            else:
                param = int(trans_param)
        else:
            param = int(
                np.clip(np.round((trans_param - d.low) / d.step) * d.step + d.low, d.low, d.high)
            )
    else:
        assert False, "Should not reach. Unexpected distribution."

    return param
