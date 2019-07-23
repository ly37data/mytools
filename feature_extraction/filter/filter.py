# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:10:22 2019

@author: liuyao
"""
import pandas as pd 

class Query():
    def __init__(self,symbol='=='):
        self.symbol = symbol
        if symbol =='=':
            raise ValueError('must "=="')
    def __call__(self,feature,value):
        if value!=None:
            self.query = f"{feature}{self.symbol}'{str(value)}'"
        else:
            self.query = feature+"!='_'"
        

__FILTER={
        "equal":Query('=='),
        '==':Query('=='),
        '<=':Query('<='),
        
        }


def Filter(filters):
    value_filter_list=[]
    value_filter_name_list=[]
    value_filter_cn_name_list=[]
    if filters!=None:
        if not isinstance(filters,list):
            raise TypeError("filters must be a list")
        for filter_ in filters:
            filters_features=filter_.get("filters_feature")
            for fil in filter_.get("filters"):
                filters_value=fil.get("filters_value",None)
                if  isinstance(filters_value,list):
                    filters_value={"==":filters_value}
                if not isinstance(filters_value,dict):
                    raise TypeError(' filters_value must be a dict  ')
                minlist=[]
                for features_,keys_,values_ in zip(filters_features,filters_value.keys(),filters_value.values()):
                    f = Query(symbol=keys_)
                    for v in values_:
                        f(features_,v)
                        minlist.append(f.query)
                                    
                value_filter_list.append(" and ".join(minlist))
                value_filter_name_list.append(fil.get("filters_name"))
                value_filter_cn_name_list.append(fil.get("filters_cn_name"))
    return value_filter_list,value_filter_name_list,value_filter_cn_name_list





