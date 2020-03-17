from datetime import datetime

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from feature_extraction import variable_types as vtypes


def infer_variable_types(df, variable_types, time_index, secondary_time_index):
    '''Infer variable types from dataframe

    Args:
        df (DataFrame): Input DataFrame
        variable_types (dict[str -> dict[str -> type]]) : An entity's
            variable_types dict maps string variable ids to types (:class:`.Variable`)
            or (type, kwargs) to pass keyword arguments to the Variable.
        time_index (str or None): Name of time_index column
        secondary_time_index (dict[str: [str]]): Dictionary of secondary time columns
            that each map to a list of columns that depend on that secondary time
    '''
    # TODO: set pk and pk types here
    inferred_types = {}
    vids_to_assume_datetime = [time_index,secondary_time_index]
    inferred_type = vtypes.Unknown
    for variable in df.columns:

        if variable in variable_types:
            continue

        elif variable in vids_to_assume_datetime:
            if col_is_datetime(df[variable]):
                inferred_type = vtypes.Datetime
            else:
                inferred_type = vtypes.Numeric

        elif df[variable].dtype == "object":
            if not len(df[variable]):
                inferred_type = vtypes.Categorical
            elif col_is_datetime(df[variable]):
                inferred_type = vtypes.Datetime
            else:
                inferred_type = vtypes.Categorical

                # heuristics to predict this some other than categorical
                sample = df[variable].sample(min(10000, len(df[variable])))

                # catch cases where object dtype cannot be interpreted as a string
                try:
                    avg_length = sample.str.len().mean()
                    if avg_length > 50:
                        inferred_type = vtypes.Text
                except AttributeError:
                    pass

        elif df[variable].dtype == "bool":
            inferred_type = vtypes.Boolean

        elif pdtypes.is_categorical_dtype(df[variable].dtype):
            inferred_type = vtypes.Categorical

        elif pdtypes.is_numeric_dtype(df[variable].dtype):
            inferred_type = vtypes.Numeric

        elif col_is_datetime(df[variable]):
            inferred_type = vtypes.Datetime

        elif len(df[variable]):
            sample = df[variable] \
                .sample(min(10000, df[variable].nunique(dropna=False)))

            unique = sample.unique()
            percent_unique = sample.size / len(unique)

            if percent_unique < .05:
                inferred_type = vtypes.Categorical
            else:
                inferred_type = vtypes.Numeric

        inferred_types[variable] = inferred_type

    return inferred_types


def convert_all_variable_data(df, variable_types):
    """Convert all dataframes' variables to different types.
    """
    for var_id, desired_type in variable_types.items():
        type_args = {}
        if isinstance(desired_type, tuple):
            # grab args before assigning type
            type_args = desired_type[1]
            desired_type = desired_type[0]

        if var_id not in df.columns:
            raise LookupError("Variable ID %s not in DataFrame" % (var_id))
        current_type = df[var_id].dtype.name

        if issubclass(desired_type, vtypes.Numeric) and \
                current_type not in vtypes.PandasTypes._pandas_numerics:
            df = convert_variable_data(df=df,
                                       column_id=var_id,
                                       new_type=desired_type,
                                       **type_args)

        if issubclass(desired_type, vtypes.Discrete) and \
                current_type not in [vtypes.PandasTypes._categorical]:
            df = convert_variable_data(df=df,
                                       column_id=var_id,
                                       new_type=desired_type,
                                       **type_args)

        if issubclass(desired_type, vtypes.Datetime) and \
                current_type not in vtypes.PandasTypes._pandas_datetimes:
            df = convert_variable_data(df=df,
                                       column_id=var_id,
                                       new_type=desired_type,
                                       **type_args)

    return df


def convert_variable_data(df, column_id, new_type, **kwargs):
    """Convert dataframe's variable to different type.
    """
    if df[column_id].empty:
        return df
    if new_type == vtypes.Numeric:
        orig_nonnull = df[column_id].dropna().shape[0]
        df[column_id] = pd.to_numeric(df[column_id], errors='coerce')
        # This will convert strings to nans
        # If column contained all strings, then we should
        # just raise an error, because that shouldn't have
        # been converted to numeric
        nonnull = df[column_id].dropna().shape[0]
        if nonnull == 0 and orig_nonnull != 0:
            raise TypeError("Attempted to convert all string column {} to numeric".format(column_id))
    elif issubclass(new_type, vtypes.Datetime):
        format = kwargs.get("format", None)
        # TODO: if float convert to int?
        df[column_id] = pd.to_datetime(df[column_id], format=format,
                                       infer_datetime_format=True)
    elif new_type == vtypes.Boolean:
        map_dict = {kwargs.get("true_val", True): True,
                    kwargs.get("false_val", False): False,
                    True: True,
                    False: False}
        # TODO: what happens to nans?
        df[column_id] = df[column_id].map(map_dict).astype(np.bool)
    elif not issubclass(new_type, vtypes.Discrete):
        raise Exception("Cannot convert column %s to %s" %
                        (column_id, new_type))
    return df


def get_linked_vars(entity):
    """Return a list with the entity linked variables.
    """
    link_relationships = [r for r in entity.entityset.relationships
                          if r.parent_entity.id == entity.id or
                          r.child_entity.id == entity.id]
    link_vars = [v.id for rel in link_relationships
                 for v in [rel.parent_variable, rel.child_variable]
                 if v.entity.id == entity.id]
    return link_vars


def col_is_datetime(col):
    # check if dtype is datetime
    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.iloc[0], datetime))):
        return True

    # if it can be casted to numeric, it's not a datetime
    dropped_na = col.dropna()
    try:
        pd.to_numeric(dropped_na, errors='raise')
    except (ValueError, TypeError):
        # finally, try to cast to datetime
        if col.dtype.name.find('str') > -1 or col.dtype.name.find('object') > -1:
            try:
                pd.to_datetime(dropped_na, errors='raise')
            except Exception:
                return False
            else:
                return True

    return False

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




