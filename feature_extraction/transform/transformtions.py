# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:02:10 2019

@author: liuyao
"""

import pandas as pd 

class Value_map():
    def __init__(self):
        self.name = ["_value_map"]
        self.cn_name = [".映射"]
        self.new_feature=[]
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,dict):
                raise ValueError(
                        '''transformation must defined like {"value_map":{"1":"a","2":"a","3":"c"}}
                        ''') 
        if len(feature)==1  :
            df[feature[0]]=df[feature[0]].astype("str")
            new_feature = feature[0]+"_value_map"
            df[new_feature] = df[feature[0]].map(tran_map)
            self.new_feature.append(new_feature)
            return df
        else:
            raise ValueError("'value_map' :features Length  must  equal to 1 ")        
        
class TimeInterval():
    def __init__(self):
        self.name = ["_timeinterval"]
        self.cn_name = [".间隔时间"]
        self.new_feature = []
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,list):
                raise ValueError(
                        '''transformation must defined like {"TimeInterval":["user_id"]}
                        ''') 
        if len(feature)==1  :
            new_feature = feature[0]+"_timeinterval"
            df = df.groupby(tran_map).apply(_TimeInterval,
                                     columns=feature[0]).reset_index(drop=True).fillna(0)
            self.new_feature.append(new_feature)
            return df
        else:
            raise ValueError("'timeinterval' :features Length  must  equal to 1 ")        

def _TimeInterval(df,columns):
    df_ = df.sort_values(columns)
    df_[columns] = pd.to_datetime(df_[columns])
    df_[columns+"_timeinterval"] = df_[columns].diff().apply(lambda x:x.days)
    return df_
        
class TimeTran():
    def __init__(self):
        self.name = []
        self.cn_name = []
        self.new_feature=[]
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,list):
            raise ValueError(
                        '''transformation must defined like {"TimeInterval":["user_id"]}
                        ''') 
        if len(feature)==1  :
            for tran_ in tran_map:
                new_feature = feature[0]+"_"+tran_
                df[feature[0]] = pd.to_datetime(df[feature[0]])
                self.new_feature.append(new_feature)
                if tran_ =="day":
                    df[new_feature] = df[feature[0]].dt.day
                    tran_name='日期'
                elif tran_ =='year':
                    df[new_feature] = df[feature[0]].dt.year
                    tran_name='年份'
                elif tran_ =='month':
                    df[new_feature] = df[feature[0]].dt.month    
                    tran_name='月份'
                elif tran_ =='weekday':
                    df[new_feature] = df[feature[0]].dt.dayofweek 
                    tran_name='星期'
                self.name.append('_'+tran_)
                self.cn_name.append('.'+tran_name)
            return df
  
class TimeDiff():
    def __init__(self):
        self.name = []
        self.cn_name = []
        self.new_feature = []
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,list):
                raise ValueError(
                        '''transformation must defined like {"timediff":["day"]}
                        ''') 
        if len(feature)==2  :
            for tran_ in tran_map:
                new_feature = feature[0]+"_"+feature[1]+"_timediff_"+tran_
                df[feature[0]] = pd.to_datetime(df[feature[0]])
                df[feature[1]] = pd.to_datetime(df[feature[1]])
                if tran_ =="day":
                    df[new_feature] = (df[feature[0]]-df[feature[0]]).dt.days
                    tran_name='天数'
                self.name.append('_timediff_'+tran_)
                self.cn_name.append('.时间差'+tran_name)
                self.new_feature.append(new_feature)
            return df
        else:
            raise ValueError("'timediff' :features Length  must  equal to 2 ")


class LessThan():
    '''
    小于一列或者小于某一个特定的值
    '''
    def __init__(self):
        self.name = []
        self.cn_name = []
        self.new_feature = []
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,list):
                raise ValueError(
                        '''transformation must defined like {"lessthan":["day"]}
                        ''') 
        if len(feature)==1  :
            for tran_ in tran_map:
                if isinstance(tran_,str) and tran_ in df.columns:
                    new_feature = feature[0]+"_lessthan_"+tran_
                    df[new_feature] = df[feature[0]]<df[tran_]
                    self.name.append("_lessthan_"+tran_)
                else:
                    new_feature = feature[0]+"_lessthan_"+tran_
                    df[new_feature] = df[feature[0]]< tran_
                    self.name.append("_lessthan_"+tran_)
                self.cn_name.append('.小于'+tran_)
                self.new_feature.append(new_feature)
            return df
        else:
            raise ValueError("'LessThan' :features Length  must  equal to 1")

class GreaterThan():
    '''
    大于一列或者小于某一个特定的值
    '''
    def __init__(self):
        self.name = []
        self.cn_name = []
        self.new_feature = []
        
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,list):
                raise ValueError(
                        '''transformation must defined like {"greaterthan":["day"]}
                        ''') 
        if len(feature)==1  :
            for tran_ in tran_map:
                if isinstance(tran_,str) and tran_ in df.columns:
                    new_feature = feature[0]+"_greaterthan_"+tran_
                    df[new_feature] = df[feature[0]]<=df[tran_]
                else:
                    new_feature = feature[0]+"_greaterthan_"+tran_
                    df[new_feature] = df[feature[0]]<= tran_
                self.name.append("_greaterthan_"+tran_)
                self.cn_name.append('.大于'+tran_)
                self.new_feature.append(new_feature)
            return df
        else:
            raise ValueError("'greaterthan' :features Length  must  equal to 1")


class EqualTo():
    '''
    等于一列或者小于某一个特定的值
    '''
    def __init__(self):
        self.name = []
        self.cn_name = []
        self.new_feature = []
        
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,list):
                raise ValueError(
                        '''transformation must defined like {"equalto":["day"]}
                        ''') 
        if len(feature)==1  :
            for tran_ in tran_map:
                if isinstance(tran_,str) and tran_ in df.columns:
                    new_feature = feature[0]+"_equalto_"+tran_
                    df[new_feature] = df[feature[0]]=df[tran_]
                else:
                    new_feature = feature[0]+"_equalto_"+tran_
                    df[new_feature] = df[feature[0]]= tran_
                self.name.append("_equalto_"+tran_)
                self.cn_name.append('.等于'+tran_)
                self.new_feature.append(new_feature)
            return df
        else:
            raise ValueError("'greaterthan' :features Length  must  equal to 1")

class NotEqualTo():
    '''
    不等于一列或者小于某一个特定的值
    '''
    def __init__(self):
        self.name = []
        self.cn_name = []
        self.new_feature = []
        
    def __call__(self,df,feature,tran_map=None):
        if not isinstance(tran_map,list):
                raise ValueError(
                        '''transformation must defined like {"notequalto":["day"]}
                        ''') 
        if len(feature)==1  :
            for tran_ in tran_map:
                if isinstance(tran_,str) and tran_ in df.columns:
                    new_feature = feature[0]+"_notequalto_"+tran_
                    df[new_feature] = df[feature[0]]=df[tran_]
                else:
                    new_feature = feature[0]+"_notequalto_"+tran_
                    df[new_feature] = df[feature[0]]= tran_
                self.name.append("_notequalto_"+tran_)
                self.cn_name.append('.不等于'+tran_)
                self.new_feature.append(new_feature)
            return df
        else:
            raise ValueError("'notequalto' :features Length  must  equal to 1")









