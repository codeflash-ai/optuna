from __future__ import annotations

import collections
from typing import Any

import optuna
from optuna._imports import try_import
from optuna.trial._state import TrialState


with try_import() as _imports:
    # `Study.trials_dataframe` is disabled if pandas is not available.
    import pandas as pd

# Required for type annotation in `Study.trials_dataframe`.
if not _imports.is_successful():
    pd = object  # NOQA

__all__ = ["pd"]


def _create_records_and_aggregate_column(
    study: "optuna.Study", attrs: tuple[str, ...]
) -> tuple[list[dict[tuple[str, str], Any]], list[tuple[str, str]]]:
    attrs_to_df_columns: dict[str, str] = {}
    for attr in attrs:
        if attr.startswith("_"):
            # Python conventional underscores are omitted in the dataframe.
            df_column = attr[1:]
        else:
            df_column = attr
        attrs_to_df_columns[attr] = df_column

    # column_agg is an aggregator of column names.
    # Keys of column agg are attributes of `FrozenTrial` such as 'trial_id' and 'params'.
    # Values are dataframe columns such as ('trial_id', '') and ('params', 'n_layers').
    column_agg: collections.defaultdict[str, set] = collections.defaultdict(set)
    non_nested_attr = ""

    metric_names = study.metric_names

    # Local binding for performance
    get_trials = study.get_trials
    directions = study.directions

    append_record = records_append = []
    records = records_append
    for trial in get_trials(deepcopy=False):
        record = {}
        trial_getattr = getattr
        for attr, df_column in attrs_to_df_columns.items():
            value = trial_getattr(trial, attr)
            if isinstance(value, TrialState):
                value = value.name
            if isinstance(value, dict):
                for nested_attr, nested_value in value.items():
                    key = (df_column, nested_attr)
                    record[key] = nested_value
                    column_agg[attr].add(key)
            elif attr == "values":
                trial_values = [None] * len(directions) if value is None else value
                iterator = (
                    enumerate(trial_values)
                    if metric_names is None
                    else zip(metric_names, trial_values)
                )
                for nested_attr, nested_value in iterator:
                    key = (df_column, nested_attr)
                    record[key] = nested_value
                    column_agg[attr].add(key)
            elif isinstance(value, list):
                for nested_attr, nested_value in enumerate(value):
                    key = (df_column, nested_attr)
                    record[key] = nested_value
                    column_agg[attr].add(key)
            elif attr == "value":
                nested_attr = non_nested_attr if metric_names is None else metric_names[0]
                key = (df_column, nested_attr)
                record[key] = value
                column_agg[attr].add(key)
            else:
                key = (df_column, non_nested_attr)
                record[key] = value
                column_agg[attr].add(key)

        records.append(record)

    columns: list[tuple[str, str]] = []
    # Use list.extend for better performance (avoids sum of lists)
    for k in attrs:
        if k in column_agg:
            columns.extend(sorted(column_agg[k]))

    return records, columns


def _flatten_columns(columns: list[tuple[str, str]]) -> list[str]:
    # Flatten the `MultiIndex` columns where names are concatenated with underscores.
    # Filtering is required to omit non-nested columns avoiding unwanted trailing underscores.
    return ["_".join(filter(lambda c: c, map(lambda c: str(c), col))) for col in columns]


def _trials_dataframe(
    study: "optuna.Study", attrs: tuple[str, ...], multi_index: bool
) -> "pd.DataFrame":
    _imports.check()

    # If no trials, return an empty dataframe.
    if len(study.get_trials(deepcopy=False)) == 0:
        return pd.DataFrame()

    if "value" in attrs and study._is_multi_objective():
        attrs = tuple("values" if attr == "value" else attr for attr in attrs)

    records, columns = _create_records_and_aggregate_column(study, attrs)

    df = pd.DataFrame(records, columns=pd.MultiIndex.from_tuples(columns))

    if not multi_index:
        df.columns = _flatten_columns(columns)

    return df
