# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:52:03 2019

@author: liuyao
"""
import pandas as pd
import numpy as np
 
import multiprocessing 
from multiprocessing import pool
import itertools
from urllib.parse import urlsplit

from .base import Variable,get_var_type
from mytools.data_exploration.config import config
from .correlations import (
    calculate_correlations,
    perform_check_correlation,
)
from .messages import (
    check_variable_messages,
    check_table_messages,
)
from mytools.data_exploration.utils import update
from mytools.data_exploration.view import plot



def describe_unsupported(series: pd.Series, series_description: dict):
    """Describe an unsupported series.

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    # number of observations in the Series
    leng = len(series)
    # number of non-NaN observations in the Series
    count = series.count()
    # number of infinte observations in the Series
    n_infinite = count - series.count()

    results_data = {
        "count": count,
        "p_missing": 1 - count * 1.0 / leng,
        "n_missing": leng - count,
        "p_infinite": n_infinite * 1.0 / leng,
        "n_infinite": n_infinite,
        "memorysize": series.memory_usage(),
    }

    return results_data


def describe_supported(series: pd.Series, series_description: dict) -> dict:
    """Describe a supported series.

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    # number of observations in the Series
    leng = len(series)
    # TODO: fix infinite logic
    # number of non-NaN observations in the Series
    count = series.count()
    # number of infinite observations in the Series
    n_infinite = count - series.count()

    # TODO: check if we prefer without nan
    distinct_count = series_description["distinct_count_with_nan"]

    stats = {
        "count": count,
        "distinct_count": distinct_count,
        "p_missing": 1 - count * 1.0 / leng,
        "n_missing": leng - count,
        "p_infinite": n_infinite * 1.0 / leng,
        "n_infinite": n_infinite,
        "is_unique": distinct_count == leng,
        "mode": series.mode().iloc[0] if count > distinct_count > 1 else series[0],
        "p_unique": distinct_count * 1.0 / leng,
        "memorysize": series.memory_usage(),
    }

    return stats


def describe_numeric_1d(series: pd.Series, series_description: dict) -> dict:
    """Describe a numeric series.

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    quantiles = config["vars"]["num"]["quantiles"].get(list)

    stats = {
        "mean": series.mean(),
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "max": series.max(),
        "kurtosis": series.kurt(),
        "skewness": series.skew(),
        "sum": series.sum(),
        "mad": series.mad(),
        "n_zeros": (len(series) - np.count_nonzero(series)),
        "histogramdata": series,
    }

    stats["range"] = stats["max"] - stats["min"]
    stats.update(
        {
            "{:.0%}".format(percentile): value
            for percentile, value in series.quantile(quantiles).to_dict().items()
        }
    )
    stats["iqr"] = stats["75%"] - stats["25%"]
    stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else np.NaN
    stats["p_zeros"] = float(stats["n_zeros"]) / len(series)

    return stats


def describe_date_1d(series: pd.Series, series_description: dict) -> dict:
    """Describe a date series.

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    stats = {
        "min": series.min(),
        "max": series.max(),
        "histogramdata": series,  # TODO: calc histogram here?
    }

    stats["range"] = stats["max"] - stats["min"]

    return stats


def describe_categorical_1d(series: pd.Series, series_description: dict) -> dict:
    """Describe a categorical series.

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    # Make sure we deal with strings (Issue #100)
    series = series.astype(str)

    # Only run if at least 1 non-missing value
    value_counts = series_description["value_counts_without_nan"]

    stats = {"top": value_counts.index[0], "freq": value_counts.iloc[0]}

    check_composition = config["vars"]["cat"]["check_composition"].get(bool)
    if check_composition:
        contains = {
            "chars": series.str.contains(r"[a-zA-Z]", case=False, regex=True).any(),
            "digits": series.str.contains(r"[0-9]", case=False, regex=True).any(),
            "spaces": series.str.contains(r"\s", case=False, regex=True).any(),
            "non-words": series.str.contains(r"\W", case=False, regex=True).any(),
        }
        stats["max_length"] = series.str.len().max()
        stats["mean_length"] = series.str.len().mean()
        stats["min_length"] = series.str.len().min()
        stats["composition"] = contains

    return stats


def describe_url_1d(series: pd.Series, series_description: dict) -> dict:
    """Describe a url series.

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    # Make sure we deal with strings (Issue #100)
    series = series[~series.isnull()].astype(str)

    stats = {}

    # Create separate columns for each URL part
    keys = ["scheme", "netloc", "path", "query", "fragment"]
    url_parts = dict(zip(keys, zip(*series.map(urlsplit))))
    for name, part in url_parts.items():
        stats["{}_counts".format(name.lower())] = pd.Series(
            part, name=name
        ).value_counts()

    # Only run if at least 1 non-missing value
    value_counts = series_description["value_counts_without_nan"]

    stats["top"] = value_counts.index[0]
    stats["freq"] = value_counts.iloc[0]

    return stats


def describe_boolean_1d(series: pd.Series, series_description: dict) -> dict:
    """Describe a boolean series.

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    value_counts = series_description["value_counts_without_nan"]

    stats = {"top": value_counts.index[0], "freq": value_counts.iloc[0]}

    return stats


def describe_constant_1d(series: pd.Series, series_description: dict) -> dict:
    """Describe a constant series (placeholder).

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        An empty dict.
    """
    return {}


def describe_unique_1d(series: pd.Series, series_description: dict) -> dict:
    """Describe a unique series (placeholder).

    Args:
        series: The Series to describe.
        series_description: The dict containing the series description so far.

    Returns:
        An empty dict.
    """
    return {}


def describe_1d(column,series):
    
    # Replace infinite values with NaNs to avoid issues with histograms later.
    series.replace(to_replace=[np.inf, np.NINF, np.PINF], value=np.nan, inplace=True)
    
    # Infer variable types
    series_description = get_var_type(series)
    
    # Run type specific analysis
    if series_description["type"] == Variable.S_TYPE_UNSUPPORTED:
        series_description.update(describe_unsupported(series, series_description))
    else:
        series_description.update(describe_supported(series, series_description))    

    type_to_func = {
        Variable.S_TYPE_CONST: describe_constant_1d,
        Variable.TYPE_BOOL: describe_boolean_1d,
        Variable.TYPE_NUM: describe_numeric_1d,
        Variable.TYPE_DATE: describe_date_1d,
        Variable.S_TYPE_UNIQUE: describe_unique_1d,
        Variable.TYPE_CAT: describe_categorical_1d,
        Variable.TYPE_URL: describe_url_1d,
    }
    
    if series_description["type"] in type_to_func:
        series_description.update(
            type_to_func[series_description["type"]](series, series_description)
        )
    else:
        raise ValueError("Unexpected type")

    # Return the description obtained
    return column,series_description


def describe_table(df: pd.DataFrame, variable_stats: pd.DataFrame) -> dict:
    """General statistics for the DataFrame.

    Args:
      df: The DataFrame to describe.
      variable_stats: Previously calculated statistic on the DataFrame.

    Returns:
        A dictionary containg the table statistics.
    """
    n = len(df)
    memory_size = df.memory_usage(index=True).sum()
    record_size = float(memory_size) / n

    table_stats = {
        "n": n,
        "nvar": len(df.columns),
        "memsize": memory_size,
        "recordsize": record_size,
        "n_cells_missing": variable_stats.loc["n_missing"].sum(),
        "n_vars_with_missing": sum((variable_stats.loc["n_missing"] > 0).astype(int)),
        "n_vars_all_missing": sum((variable_stats.loc["n_missing"] == n).astype(int)),
    }

    table_stats["p_cells_missing"] = table_stats["n_cells_missing"] / (
        table_stats["n"] * table_stats["nvar"]
    )

    supported_columns = variable_stats.transpose()[
        variable_stats.transpose().type != Variable.S_TYPE_UNSUPPORTED
    ].index.tolist()
    table_stats["n_duplicates"] = (
        sum(df.duplicated(subset=supported_columns))
        if len(supported_columns) > 0
        else 0
    )
    table_stats["p_duplicates"] = (
        (table_stats["n_duplicates"] / len(df))
        if (len(supported_columns) > 0 and len(df) > 0)
        else 0
    )

    # Variable type counts
    table_stats.update({k.value: 0 for k in Variable})
    table_stats.update(
        dict(variable_stats.loc["type"].apply(lambda x: x.value).value_counts())
    )
    table_stats[Variable.S_TYPE_REJECTED.value] = (
        table_stats[Variable.S_TYPE_CONST.value]
        + table_stats[Variable.S_TYPE_CORR.value]
        + table_stats[Variable.S_TYPE_RECODED.value]
    )
    return table_stats


def get_missing_diagrams(df: pd.DataFrame, table_stats: dict) -> dict:
    """Gets the rendered diagrams for missing values.

    Args:
        table_stats: The overall statistics for the DataFrame.
        df: The DataFrame on which to calculate the missing values.

    Returns:
        A dictionary containing the base64 encoded plots for each diagram that is active in the config (matrix, bar, heatmap, dendrogram).
    """
    missing_map = {
        "matrix": {"func": plot.missing_matrix, "min_missing": 0},
        "bar": {"func": plot.missing_bar, "min_missing": 0},
        "heatmap": {"func": plot.missing_heatmap, "min_missing": 2},
        "dendrogram": {"func": plot.missing_dendrogram, "min_missing": 1},
    }

    missing = {}
    for name, settings in missing_map.items():
        if (
            config["missing_diagrams"][name].get(bool)
            and table_stats["n_vars_with_missing"] >= settings["min_missing"]
        ):
            if name != "heatmap" or (
                table_stats["n_vars_with_missing"] - table_stats["n_vars_all_missing"]
                >= settings["min_missing"]
            ):
                missing[name] = settings["func"](df)
    return missing


def describe(df) -> dict:
    """Calculate the statistics for each series in this DataFrame.

    Args:
        df: DataFrame.

    Returns:
        This function returns a dictionary containing:
            - table: overall statistics.
            - variables: descriptions per series.
            - correlations: correlation matrices.
            - missing: missing value diagrams.
            - messages: direct special attention to these patterns in your data.
    """
    if not isinstance(df,pd.DataFrame):
        raise TypeError("df must be type of pd.DataFrame")       
    if df.empty:
        raise ValueError('df cannot be empty')    
    if  not pd.Index(np.arange(0,df.shape[0])).equals(df.index):
        df.reset_index(inplace=True)
    
    mp = config["pool_size"].get(int)

    if mp ==1:
        items = [(column,series) for column,series in df.iteritems()]
        series_description = {
                column:series for column,series in itertools.starmap(describe_1d,items)
                }
    else:           
        if mp<=0:
            mp = multiprocessing.cpu_count()
        with pool.ThreadPool(mp) as executor:
            series_description = {}
            results = executor.starmap(describe_1d,df.iteritems())
            for col,descri in results:
                series_description[col] = descri
            
    # Mapping from column name to variable type
    variables = {
        column: description["type"]
        for column, description in series_description.items()
    }

    # Get correlations
    correlations = calculate_correlations(df, variables)

    # Check correlations between numerical variables
    if (
        config["check_correlation_pearson"].get(bool) is True
        and "pearson" in correlations
    ):
        # Overwrites the description with "CORR" series
        correlation_threshold = config["correlation_threshold_pearson"].get(float)
        update(
            series_description,
            perform_check_correlation(
                correlations["pearson"],
                lambda x: x > correlation_threshold,
                Variable.S_TYPE_CORR,
            ),
        )

    # Check correlations between categorical variables
    if (
        config["check_correlation_cramers"].get(bool) is True
        and "cramers" in correlations
    ):
        # Overwrites the description with "CORR" series
        correlation_threshold = config["correlation_threshold_cramers"].get(float)
        update(
            series_description,
            perform_check_correlation(
                correlations["cramers"],
                lambda x: x > correlation_threshold,
                Variable.S_TYPE_CORR,
            ),
        )

    # Check recoded
    if config["check_recoded"].get(bool) is True and "recoded" in correlations:
        # Overwrites the description with "RECORDED" series
        update(
            series_description,
            perform_check_correlation(
                correlations["recoded"], lambda x: x == 1, Variable.S_TYPE_RECODED
            ),
        )

    # Transform the series_description in a DataFrame
    variable_stats = pd.DataFrame(series_description)

    # Table statistics
    table_stats = describe_table(df, variable_stats)

    # missing diagrams
    missing = get_missing_diagrams(df, table_stats)

    # Messages
    messages = check_table_messages(table_stats)
    for col, description in series_description.items():
        messages += check_variable_messages(col, description)

    return {
        # Overall description
        "table": table_stats,
        # Per variable descriptions
        "variables": series_description,
        # Correlation matrices
        "correlations": correlations,
        # Missing values
        "missing": missing,
        # Warnings
        "messages": messages,
    }







