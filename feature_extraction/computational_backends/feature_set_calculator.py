import warnings
from functools import partial

import numpy as np
import pandas as pd

from feature_extraction import variable_types
from feature_extraction.exceptions import UnknownFeature
from feature_extraction.feature_base import (
    AggregationFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)


class FeatureSetCalculator(object):
    """
    Calculates the values of a set of features for given instance ids.
    """

    def __init__(self, entity, feature_set,features):
        """
        Args:
            entity (Entity) :
                
            feature_set (FeatureSet): 

            features (Featurebase):
                
        return:
            pd.DataFrame
        """
        self.entity = entity
        self.feature_set = feature_set
        self.features = features
        self.calc_feature_names = [f.get_feature_names() for f in features]
        self.fillna_dict = {}
        
        self.num_features = len(feature_set.features_by_name.values())
        
    def run(self, frame, progress_callback=None):
        """
        Calculate values of features for the given instances of the target
        entity.

        Args:
            frame (np.ndarray or pd.Categorical): Instance ids for which
                to build features.

            progress_callback (callable): function to be called with incremental progress updates

        Returns:
            pd.DataFrame : Pandas DataFrame of calculated feature values.
                Indexed by instance_ids. Columns in same order as features
                passed in.
        """
        df = self.entity.df
        matrix = self._calculate_features(df,frame,self.features,progress_callback)
        
        return matrix
    def _calculate_features(self, df, frame, features, progress_callback):
        # Group the features so that each group can be calculated together.
        # The groups must also be in topological order (if A is a transform of B
        # then B must be in a group before A).
        feature_groups = self.feature_set.group_features()
        frame_list = [frame] 
        for group in feature_groups:
            representative_feature = group[0]
            handler = self._feature_type_handler(representative_feature)
            frame_list,df = handler(group, frame_list,df , progress_callback)
         
        frame = pd.concat(frame_list,axis=1,sort=False)
        frame.fillna(self.fillna_dict, inplace=True)
        return frame

    def _feature_type_handler(self, f):
        if type(f) == TransformFeature:
            return self._calculate_transform_features
        elif type(f) == GroupByTransformFeature:
            return self._calculate_groupby_features
        elif type(f) == AggregationFeature:
            return self._calculate_agg_features
        elif type(f) == IdentityFeature:
            return self._calculate_identity_features
        else:
            raise UnknownFeature(u"{} feature unknown".format(f.__class__))

    def _calculate_identity_features(self, features, frame, base_frame, progress_callback):
        for f in features:
            assert f.get_name() in base_frame, (
                'Column "%s" missing frome dataframe' % f.get_name())

        progress_callback(len(features) / float(self.num_features))

        return frame,base_frame

    def _calculate_transform_features(self, features, frame_list, base_frame,progress_callback):
        for f in features:
            # handle when no data
            if base_frame.shape[0] == 0:
                set_default_column(base_frame, f)

                progress_callback(1)

                continue

            # collect only the variables we need for this transformation
            variable_data = [base_frame[bf.get_name()]
                             for bf in f.base_features]

            feature_func = f.get_function()
            # apply the function to the relevant dataframe slice and add the
            # feature row to the results dataframe.
            values = feature_func(*variable_data)

            # if we don't get just the values, the assignment breaks when indexes don't match
            if f.number_output_features > 1:
                values = [strip_values_if_series(value) for value in values]
            else:
                values = [strip_values_if_series(values)]

            update_feature_columns(f, base_frame, values)

            progress_callback(1 )

        return frame_list,base_frame

    def _calculate_groupby_features(self, features, frame, _df_trie, progress_callback):
        for f in features:
            set_default_column(frame, f)

        # handle when no data
        if frame.shape[0] == 0:
            progress_callback(len(features) / float(self.num_features))

            return frame

        groupby = features[0].groupby.get_name()
        grouped = frame.groupby(groupby)
        groups = frame[groupby].unique()  # get all the unique group name to iterate over later

        for f in features:
            feature_vals = []
            for _ in range(f.number_output_features):
                feature_vals.append([])

            for group in groups:
                # skip null key if it exists
                if pd.isnull(group):
                    continue

                column_names = [bf.get_name() for bf in f.base_features]
                # exclude the groupby variable from being passed to the function
                variable_data = [grouped[name].get_group(group) for name in column_names[:-1]]
                feature_func = f.get_function()

                # apply the function to the relevant dataframe slice and add the
                # feature row to the results dataframe.
                if f.primitive.uses_calc_time:
                    values = feature_func(*variable_data, time=self.time_last)
                else:
                    values = feature_func(*variable_data)

                if f.number_output_features == 1:
                    values = [values]

                # make sure index is aligned
                for i, value in enumerate(values):
                    if isinstance(value, pd.Series):
                        value.index = variable_data[0].index
                    else:
                        value = pd.Series(value, index=variable_data[0].index)
                    feature_vals[i].append(value)

            if any(feature_vals):
                assert len(feature_vals) == len(f.get_feature_names())
                for col_vals, name in zip(feature_vals, f.get_feature_names()):
                    frame[name].update(pd.concat(col_vals))

            progress_callback(1 )

        return frame

    def _calculate_agg_features(self, features, frame_list, base_frame, progress_callback):
        test_feature = features[0]
        # handle where
        where = test_feature.where
        if where is not None and not base_frame.empty:            
            wfunc = where.get_function()
            base_frame_ = base_frame[wfunc(base_frame)].copy()
        else:
            base_frame_ = base_frame.copy()
        # handel time_window
        time_window = test_feature.time_window
        if time_window is not None and not base_frame_.empty:
            base_frame_ = base_frame_[test_feature.entity._handle_time(base_frame_,time_window)]
        # when no child data, just add all the features to frame with nan
#        if base_frame_.empty:
#            for f in features:
#                frame[f.get_name()] = np.nan
#                progress_callback(1 )
#                frame_list.append(frame)
#        else:
        groupby_var = test_feature.entity.index

        to_agg = {}
        agg_rename = {}
        to_apply = set()
        dummy_c = set()
        # apply multivariable and time-dependent features as we find them, and
        # save aggregable features for later
        for f in features:
            if _can_agg(f):

                variable_id = f.base_features[0].get_name()
                if variable_id not in to_agg:
                    to_agg[variable_id] = []
                func = f.get_function()

                funcname = func
                if callable(func):
                    # if the same function is being applied to the same
                    # variable twice, wrap it in a partial to avoid
                    # duplicate functions
                    funcname = str(id(func))
                    if u"{}-{}".format(variable_id, funcname) in agg_rename:
                        func = partial(func)
                        funcname = str(id(func))

                    func.__name__ = funcname

                to_agg[variable_id].append(func)
                # this is used below to rename columns that pandas names for us
                agg_rename[u"{}-{}".format(variable_id, funcname)] = f.get_name()
                continue
            if f.primitive.use_alone:
                dummy_c.add(f)
                
                continue
            to_apply.add(f)

        # Apply the non-aggregable functions generate a new dataframe, and merge
        # it with the existing one
        if len(to_apply):
            wrap = agg_wrapper(to_apply)
            # groupby_var can be both the name of the index and a column,
            # to silence pandas warning about ambiguity we explicitly pass
            # the column (in actuality grouping by both index and group would
            # work)
            to_merge = base_frame_.groupby(groupby_var).apply(wrap)
            frame_list.append(to_merge)

            progress_callback(len(to_apply))

        # Apply the aggregate functions to generate a new dataframe, and merge
        # it with the existing one
        if len(to_agg):
            # groupby_var can be both the name of the index and a column,
            # to silence pandas warning about ambiguity we explicitly pass
            # the column (in actuality grouping by both index and group would
            # work)
            to_merge = base_frame_.groupby(groupby_var).agg(to_agg)

            # rename columns to the correct feature names
            to_merge.columns = [agg_rename["-".join(x)] for x in to_merge.columns.ravel()]
            to_merge = to_merge[list(agg_rename.values())]
            for f in features:
                if (f.number_output_features == 1 and f.primitive.uses_calc_time ==False and
                        f.primitive.use_alone ==False and
                        f.variable_type == variable_types.Numeric and
                        to_merge[f.get_name()].dtype.name in ['object', 'bool']):
                    to_merge[f.get_name()] = to_merge[f.get_name()].astype(float)
                if f.primitive.name == 'Count' and f.primitive.replace_value is not None:
                    to_merge[f.get_name()] = to_merge[f.get_name()].replace(f.primitive.replace_value,np.nan)
            # workaround for pandas bug where categories are in the wrong order
            # see: https://github.com/pandas-dev/pandas/issues/22501
            frame_list.append(to_merge)

            # determine number of features that were just merged
            progress_callback(len(to_merge.columns))
        if len(dummy_c):
            for f in dummy_c:
                variable_id = f.base_features[0].get_name()
                func = f.primitive.get_function()
                to_merge = f.get_function()(base_frame_,groupby_var,variable_id)
                to_merge.columns = f.get_feature_names()
                frame_list.append(to_merge)
            progress_callback(len(dummy_c))

        # Handle default values
        
        for f in features:
            feature_defaults = {name: f.default_value
                                for name in f.get_feature_names()}
            self.fillna_dict.update(feature_defaults)


        # convert boolean dtypes to floats as appropriate
        # pandas behavior: https://github.com/pydata/pandas/issues/3752


        return frame_list,base_frame



def _can_agg(feature):
    assert isinstance(feature, AggregationFeature)
    base_features = feature.base_features
    if feature.where is not None:
        base_features = [bf.get_name() for bf in base_features
                         if bf.get_name() != feature.where.get_name()]

    if feature.primitive.uses_calc_time or feature.primitive.use_alone :
        return False
    single_output = feature.primitive.number_output_features == 1
    return len(base_features) == 1 and single_output


def agg_wrapper(feats):
    def wrap(df):
        d = {}
        for f in feats:
            func = f.get_function()
            variable_ids = [bf.get_name() for bf in f.base_features]
            args = [df[v] for v in variable_ids]

            values = func(*args)

            if f.number_output_features == 1:
                values = [values]
            update_feature_columns(f, d, values)

        return pd.Series(d)
    return wrap


def set_default_column(frame, f):
    for name in f.get_feature_names():
        frame[name] = f.default_value


def update_feature_columns(feature, data, values):
    names = feature.get_feature_names()
    assert len(names) == len(values)
    if isinstance(data,pd.DataFrame):
        assert data.shape[0] == values[0].shape[0]
    for name, value in zip(names, values):
        data[name] = value


def strip_values_if_series(values):
    if isinstance(values, pd.Series):
        values = values.values
    return values
