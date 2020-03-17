# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 21:47:47 2019

后续:
    1.优化聚合函数
    2.加入并行

@author: liuyao
"""


import pandas as pd
import os
from datetime import datetime
import logging
from feature_extraction.synthesis.deep_feature_synthesis import DeepFeatureSynthesis
from feature_extraction.utils import entry_point,save_df
from feature_extraction.entityset import Entity
from feature_extraction.computational_backends.calculate_feature_matrix import calculate_feature_matrix
from feature_extraction.utils.entity_utils import reduce_mem_usage
from feature_extraction.config_init import initialize_logging


@entry_point('feature_extraction_dfs')
def dfs(data=None,
        index=None,
        time_index=None,
        filters=None,
        feature_ent=None,
        prefix_name=None,
        prefix_cn_name=None,
        config=None,
        verbose=None,
        features_only=None,
        cn_map=None,
        reduce_memory=False,
        log_dir=None,
        result=None,
        ):
    '''
        data: (pd.DataFrame)
        index: (str)
        time_index: (str)
        filters: (dict)
        feature_ent: (dict)
        prefix_name: (str)
        prefix_cn_name: (str)
        config: (dict{})
        verbose: (bool)
        cn_map: (str)  path to save feature map 
        reduce_memory: (bool) default False, if True reduce the memory of result .
        log_dir: (str) 
    '''
    if log_dir is None:
        initialize_logging()    
    else:
        log_path = os.path.join(log_dir
                                ,""+datetime.now().strftime('%Y%m%d')+r'.log' if prefix_name is None 
                                else prefix_name+'_'+datetime.now().strftime('%Y%m%d')+r'.log')
        initialize_logging(log_path)
    logger = logging.getLogger('feature_extraction')

    if isinstance(config,dict):
        pass
    else:
        config={}
        config['index']=index
        config['time_index']=time_index
        config['filters']=filters
        config['feature_ent']=feature_ent
        
    entity = Entity(data,config=config,
                          prefix_name = prefix_name,
                          prefix_cn_name = prefix_cn_name)

    dfs_o = DeepFeatureSynthesis(entity)

    features,cn_map_df = dfs_o.build_features(verbose=verbose)

    if features_only:
        return features
    

    feature_matrix = calculate_feature_matrix(features,entity=entity,verbose=verbose)
    
    if reduce_memory:
        feature_matrix = reduce_mem_usage(feature_matrix,verbose=verbose)
        
    if cn_map:
        save_df(cn_map_df,cn_map)
        logger.info('Output result chinese map:{}'.format(cn_map))
        
    if result is not None:
        save_df(feature_matrix,result)
        logger.info('Output result :{}'.format(result))
        
    else:    
        return  feature_matrix

