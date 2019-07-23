# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:16:57 2019

@author: liuyao
"""
from functools import reduce
import pandas as pd 

agg_name_map={
        "count":"统计",
        "sum":"总和",
        "max":'最大值',
        'min':'最小值',
        'std':'均值',
        'mode':'众数',
        'skew':'',
        
        }
apply_name_map={
        "value_count":"组内计数",
        }



aggregation_func_ = pd.DataFrame()


def list_combination(lists,code=""):
    def func1(x,y):
        if x==[] :
            return y
        if y==[] :
            return x
        return [str(i)+code+str(j) for i in x for j in y]
    
    cb_list=reduce(func1,lists)
    return [x.replace(code+code,code).strip(".") for x in cb_list]

def list_rep(a,n):
    b=[]
    for i in a:
        for j in range(n):
            b.append(i)
    return b

def get_transfrom_list():
    transfrom_list=["value_map"]
    return transfrom_list


def _To_datetime(series):
    series = series.astype(str)
    __len = series.str.len().max()
    
    if __len<10:
        raise ValueError("c")
        
    elif __len>18:
        return pd.to_datetime(series.str[:10])
    else:
        return pd.to_datetime(series)




