# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:03:21 2019

@author: liuyao
"""

from .transformtions import Value_map,TimeInterval,TimeTran,TimeDiff,LessThan,GreaterThan,EqualTo,NotEqualTo

__TRANSFORM={
        "value_map":Value_map,
        'timeinterval':TimeInterval,
        'timetran':TimeTran,
        'timediff':TimeDiff,
        'lessthan':LessThan,
        'greaterthan':GreaterThan,
        'equalto':EqualTo,
        'notequalto':NotEqualTo,
        }
 
    
def Transform(df,transformation,feature,feature_name,feature_cn_name):
    trans_feature_list = []
    trans_feature_name = []
    trans_feature_cn_name = []
    if isinstance(transformation,dict):
        for key in transformation.keys():            
            t =   __TRANSFORM.get(key,None)()            
            if t == None:
                raise ValueError(
                        '''transformation :["value_map",'timeinterval','timetran']
                        ''')
               
            df=t(df,feature,tran_map=transformation[key])
            trans_feature_list.extend(t.new_feature)
            trans_feature_name.extend([feature_name+name for name in t.name])
            trans_feature_cn_name.extend([feature_cn_name+cn_name for cn_name in t.cn_name])            
    return df,trans_feature_list,trans_feature_name,trans_feature_cn_name

    




