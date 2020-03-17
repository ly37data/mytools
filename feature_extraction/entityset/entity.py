
import logging
from builtins import object


import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal, is_numeric_dtype,is_datetime64_any_dtype
import operator

from feature_extraction.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)
from feature_extraction import primitives

from feature_extraction import variable_types as vtypes
from feature_extraction.utils.entity_utils import (
    col_is_datetime,
    convert_all_variable_data,
    convert_variable_data,
    infer_variable_types
)
from feature_extraction.utils.wrangle import (
    _check_time_type,
    _check_timedelta,
    _dataframes_equal
)


pd.options.mode.chained_assignment = None  # default='warn'
logger = logging.getLogger('feature_extraction.entity')


class Entity(object):
    """
    Stores all actual data for a entityset

    Attributes:
        data
        config

    Properties:
        metadata

    """
    time_index = None
    
    def __init__(self, df=None,config=None,prefix_name=None,prefix_cn_name=None,variable_types=None):
        """Creates Entity

            Args:
                data (DataFrame) : 
                
                config(dict) :
                    
                prefix_name(str) :
                    
                prefix_cn_name(str) :

            Example:

        """
        if isinstance(df,pd.DataFrame):
            self.df = df.copy()
            logger.info('data shape {}'.format(self.df.shape))
        else:
            raise ValueError('df must be pandas.DataFrame')
            
        assert len(df.columns) == len(set(df.columns)), "Duplicate column names"

        self.columns = df.columns.tolist()

        self.config = config
        self.prefix_name = prefix_name
        self.prefix_cn_name = prefix_cn_name
        self.index = config.get('index',None)
        logger.info('config index loaded: {}'.format(self.index))
        
        time_index = config.get('time_index',{})
        self._check_time_index(time_index)
        
        self._create_variables(variable_types,self.index,self.time_index,self.time_cal)
        logger.info('create variables variable types ')
        self.where = []
        filters=config.get("filters",None)  
        if filters:
            filters_features = filters.get('feature',None)
            self._check_columns(filters_features)
            filters=filters.get('filter_entries',None)
            if filters is None:
                raise ValueError('must difne filters')
            else:
                
                for filter_ in filters:
                    filterbase = filter_.get('condition')
                    filterbase.set_arg(filters_features,filter_.get('condition_name_prefix',None)
                    ,filter_.get('cn_desc',None))
                    self.where.append(filterbase)
        else:
            self.where=[None]
        self.feature_ents = config.get("feature_entries")
        self.ent_features = []
        self.feature_ents_base = []
        if isinstance(self.feature_ents,list):
            for feature_ent in self.feature_ents:    
                self.feature_ents_base.append(FeatureConfigBase(feature_ent))
                feature = feature_ent.get('feature',None)
                if isinstance(feature,str):
                    self.ent_features.append(feature)
                elif isinstance(feature,list):
                    self.ent_features.extend(feature)
                else:
                    raise ValueError('feature_ents feature can not be None')
            self._check_columns(self.ent_features)
        else:
            raise TypeError('feature_ents must be a list')
        
        
    def _check_columns(self,col_list):
        
        col_list_check = [column for column in col_list if column and column not in self.columns]
        
        if len(col_list_check)>0:
            raise LookupError('{} not found in dataframe '.format(col_list_check))
    
    def _check_time_index(self,time_index):
        
        self.time_index=time_index.get("retro_index",None)
        self.time_cal = time_index.get("cur_index",None)
        
        self._check_columns([self.time_index,self.time_cal])
        
        time_window = time_index.get('time_window',[None])
        time_type = time_index.get('type',None)
        logger.info('config time_window loaded:{}'.format(time_window))

        if time_type ==None:
            time_type = 'd'

        time_type = time_type.lower()
        if time_type not in ['d','m','y']:
            raise ValueError("time_type must in ('d','m','y')")
        if time_type =='d':
            self.time_filter_name = 'Days'
            self.time_filter_cn_name = '天'

        if None in time_window:
            time_window.remove(None)
            time_window.sort()
            time_window.append(None)
        else:
            time_window.sort()
        self.time_window = time_window
        
        if self.time_index!=None and is_datetime64_any_dtype(self.df[self.time_index])==False:
            print("{} not datetime64 ".format(self.time_index))
            self.df[self.time_index]=_To_datetime(self.df[self.time_index])
        if self.time_cal!=None and is_datetime64_any_dtype(self.df[self.time_cal])==False:
            print("{} not datetime64 ".format(self.time_cal))
            self.df[self.time_cal]=_To_datetime(self.df[self.time_cal])
            
        if 'deal_time' not in self.columns:                
            if operator.eq(self.time_window,[None]):
                self.df["deal_time"] = 0
            else:
                self.df["deal_time"] = (self.df[self.time_index] - self.df[self.time_cal]).dt.days
                print("timeintel max: {} ;min  {}".format(self.df["deal_time"].max(),self.df["deal_time"].min()))

    def _handle_time(self, df,time_window=None):
        """
        Filter a dataframe for all instances before time_last.
        If this entity does not have a time index, return the original
        dataframe.
        """
        if self.time_index:
            if time_window is not None:
                mask = df['deal_time'] <=time_window
            else:
                logger.warning(
                    "time_window is None"
                )
            return mask

    def _create_variables(self, variable_types, index, time_index, secondary_time_index):
        """Extracts the variables from a dataframe

        Args:
            variable_types (dict[str -> dict[str -> type]]) : An entity's
                variable_types dict maps string variable ids to types (:class:`.Variable`)
                or (type, kwargs) to pass keyword arguments to the Variable.
            index (str): Name of index column
            time_index (str or None): Name of time_index column
            secondary_time_index (dict[str: [str]]): Dictionary of secondary time columns
                that each map to a list of columns that depend on that secondary time
        """
        variables = []
        variable_types = variable_types.copy() if variable_types is not None else {}
        for index_ in index:
            if index_ not in variable_types:
                variable_types[index_] = vtypes.Index

        inferred_variable_types = infer_variable_types(self.df,
                                                       variable_types,
                                                       time_index,
                                                       secondary_time_index=secondary_time_index)
        inferred_variable_types.update(variable_types)
        
        for v in inferred_variable_types:
            # TODO document how vtype can be tuple
            vtype = inferred_variable_types[v]
            if isinstance(vtype, tuple):
                # vtype is (ft.Variable, dict_of_kwargs)
                _v = vtype[0](v, self, **vtype[1])
            else:
                _v = inferred_variable_types[v](v, self)
            variables += [_v]
        # convert data once we've inferred
        self.df = convert_all_variable_data(df=self.df,
                                            variable_types=inferred_variable_types)
        # make sure index is at the beginning
        index_variable = [v for v in variables
                          if v.id in index][0]
        self.variables = [index_variable] + [v for v in variables
                                             if v.id not in  index]


class FeatureConfigBase(object):
    
    def __init__(self,feature_ent):
        self.feature = feature_ent.get('feature',None)
        
        trans_primitives = feature_ent.get('preprocessing',None)
        self.trans_primitives=[]
        if not isinstance(trans_primitives,list):
            if isinstance(trans_primitives,str) or isinstance(trans_primitives,PrimitiveBase):
                trans_primitives = [trans_primitives]               
        if trans_primitives is not None:                         
            for t in trans_primitives:
                t = check_trans_primitive(t)
                self.trans_primitives.append(t)
            
        agg_primitives = feature_ent.get('aggregating',None)
        self.agg_primitives=[]
        if not isinstance(agg_primitives,list):
            if isinstance(agg_primitives,str) or isinstance(agg_primitives,PrimitiveBase) :
                agg_primitives = [agg_primitives]
        if agg_primitives is not None:                         
            for t in agg_primitives:
                t = check_agg_primitive(t)
                self.agg_primitives.append(t)  
                
        self.feature_name = feature_ent.get('feature_name_prefix',None)
        self.feature_cn_name = feature_ent.get('cn_desc',None)

def _To_datetime(series):
    series = series.astype(str)
    __len = series.str.len().max()
            
    if __len>18:
        return pd.to_datetime(series.str[:10])
    else:
        return pd.to_datetime(series)


def handle_primitive(primitive):
    if not isinstance(primitive, PrimitiveBase):
        primitive = primitive()
    assert isinstance(primitive, PrimitiveBase), "must be a primitive"
    return primitive


def check_trans_primitive(primitive):
    trans_prim_dict = primitives.get_transform_primitives()

    if isinstance(primitive,str):
        if primitive.lower() not in trans_prim_dict:
            raise ValueError("Unknown transform primitive {}. ".format(primitive),
                             "Call ft.primitives.list_primitives() to get",
                             " a list of available primitives")
        primitive = trans_prim_dict[primitive.lower()]
    primitive = handle_primitive(primitive)
    if not isinstance(primitive, TransformPrimitive):
        raise ValueError("Primitive {} in trans_primitives or "
                         "groupby_trans_primitives is not a transform "
                         "primitive".format(type(primitive)))
    return primitive

def check_agg_primitive(primitive):
    agg_prim_dict = primitives.get_aggregation_primitives()

    if isinstance(primitive,str):
        if primitive.lower() not in agg_prim_dict:
            raise ValueError("Unknown aggregation primitive {}. ".format(primitive),
                             "Call ft.primitives.list_primitives() to get",
                             " a list of available primitives")
        primitive = agg_prim_dict[primitive.lower()]
    primitive = handle_primitive(primitive)
    if not isinstance(primitive, AggregationPrimitive):
        raise ValueError("Primitive {} in agg_primitives is not an "
                             "aggregation primitive".format(type(primitive)))
    return primitive
