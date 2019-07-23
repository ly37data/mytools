# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:15:14 2019
synthesis
1. 增加transform:图谱特征以及w2v特征
2. 可以处理多条config:clear
3. 两层衍生: 时序的变动情况。3个月和6个月的比值。
4. 和pandas_pro iV 计算一起使用。

@author: liuyao
"""


import pandas as pd 
import numpy as np
from pandas.api.types import is_datetime64_any_dtype
import warnings
from .utils import agg_name_map,list_combination,list_rep,_To_datetime
from .transform import Transform
from .filter import Filter
import time
import operator
from collections import namedtuple
from itertools import combinations

domain = namedtuple('domain',['index','feature','filters','aggregation','deep'])

class Synthesis():
    def __init__(self,config):
        self.config = config
        self.filter_name_list=[]
        self.filter_cn_name_list=[]
        self.columns_name_cn = pd.DataFrame(columns=["feature_name","cn_name"])
        self.filter_name_list=[]
        self.filter_cn_name_list=[]
        self.df = pd.DataFrame()
        self.config_ = []
        self.config_list = []
    
    def dfs_1d(self,target_):
        agg_map_list = [i for i in target_[3] if i in list(agg_name_map.keys())]
        apl_map_list = [i for i in target_[3] if i in ["value_counts"]]
        print(self.df.query(target_[0]).shape)
        if len(agg_map_list)>0:
            gb_re=self.df.query(target_[0]).groupby(target_[1]).agg({target_[2]:agg_map_list})
            gb_re.columns = [target_[4]+"_"+x for x in agg_map_list]
            
            name_cn = [target_[5]+"."+agg_name_map.get(x) for x in agg_map_list]
            name_cn_df=pd.DataFrame(np.array([gb_re.columns,name_cn]).T,
                                         columns=["feature_name","cn_name"])

        if len(apl_map_list)>0:
            for apl_map in apl_map_list:
                if apl_map=='value_counts':
                    gb_re=self.df.query(target_[0]).groupby(
                            target_[1])[target_[2]].apply(pd.value_counts
                                   ).unstack()
                name_cn = [target_[5]+ '.'+x+".组内计数" for x in gb_re.columns.tolist()]
                gb_re.columns = [target_[4]+"_"+x+"_valuecounts" for x in gb_re.columns.tolist()]
            
                name_cn_df=pd.DataFrame(np.array([gb_re.columns,name_cn]).T,
                                         columns=["feature_name","cn_name"])

        return gb_re,name_cn_df
            
    def dfs(self):
        
        print("--- start dfs ---")
        _start_time = time.time()
        
        
        ls_config = len(self.config)
        if ls_config==0:
            raise ValueError("config is empty")

        #df["deal_time"] = df["deal_time"].apply(lambda x:x.days)
        result=[]
        name_cn_list=[]
        for val in self.config:
            self.config_s(val)                    
        for target_ in self.config_:
            gb_re,name_cn_df = self.dfs_1d(target_)
            result.append(gb_re)
            name_cn_list.append(name_cn_df) 
                    
        feature_result = pd.concat(result,axis=1).reset_index()
        feature_cn_map = pd.concat(name_cn_list,ignore_index=True)
        
        print("--- create {} features: in {}s ---".format(feature_cn_map.shape[0],
              round(time.time()-_start_time,2)))
        
        return  feature_result,feature_cn_map
    
    def _get_data(self,val):
        df = val.get("data",None)
        if isinstance(df,pd.DataFrame):
            self.df = df.copy(deep=True)
        elif isinstance(df,str):
            self.df = pd.read_csv(self.df)

        self.columns_list = self.df.columns.tolist()
            
               
    
    def _check_time_index(self,time_index):
        
        self.time_index=time_index.get("time_index",None)
        self.time_cal = time_index.get("time_cal",None)
        self.time_window = time_index.get('time_window',[None])
        
        
        if self.time_index!=None and is_datetime64_any_dtype(self.df[self.time_index])==False:
            print("{} not datetime64 dtype".format(self.time_index))
            self.df[self.time_index]=_To_datetime(self.df[self.time_index])
        if self.time_cal!=None and is_datetime64_any_dtype(self.df[self.time_cal])==False:
            print("{} not datetime64 dtype".format(self.time_cal))
            self.df[self.time_cal]=_To_datetime(self.df[self.time_cal]) 
        if operator.eq(self.time_window,[None]):
            self.df["deal_time"] = 0
        else:
            self.df["deal_time"] = (self.df[self.time_index] - self.df[self.time_cal]).dt.days
            
    def _check_feature_columns(self):
        pass
           
    def config_s(self,val):
        
        self._get_data(val)
        
        target=val.get("index",None)
        
        '''
        1 取时间窗口配置文件
        '''
        time_index=val.get("time_index",{})
        
        time_filter_list = []
        time_filter_name_list = []
        time_filter_cn_name_list = []
        self._check_time_index(time_index)        
                    
        time_filter_list=list(map(lambda x:"0<deal_time<="+str(x) if x!=None else "deal_time<=inf",self.time_window))
        time_filter_name_list=list(map(lambda x:str(x)+"d" if x!=None else "",self.time_window))
        time_filter_cn_name_list=list(map(lambda x:"近"+str(x)+"天" if x!=None else "",self.time_window))    
                
        
        '''
        2 样本特征过滤配置文件
        '''
        filters=val.get("filters",None)
        value_filter_list,value_filter_name_list,value_filter_cn_name_list = Filter(filters)
                
        filter_list = list_combination([value_filter_list,time_filter_list],code=" and ")
        
        #特征名称
        prefix_name =  val.get("prefix_name",None)
        prefix_cn_name =  val.get("prefix_cn_name",None)        
        if prefix_name ==None:
            filter_name_list = list_combination([value_filter_name_list,
                                                 time_filter_name_list],code="_")
            filter_cn_name_list = list_combination([value_filter_cn_name_list,
                                                    time_filter_cn_name_list],code=".")
        elif prefix_cn_name==None:
            raise ValueError('prefix_name is not empty but prefix_cn_name is empty')
        else:
            if isinstance(prefix_name,str) and isinstance(prefix_cn_name,str):
                filter_name_list = list_combination([[prefix_name],value_filter_name_list,
                                                     time_filter_name_list],code="_")
                filter_cn_name_list = list_combination([[prefix_cn_name],value_filter_cn_name_list,
                                                        time_filter_cn_name_list],code=".")
        
        '''
        3 特征计算配置文件
        '''        
        feature_ents = val.get("feature_ent")
        
        feature_list=[]
        feature_aggmap_list=[]
        feature_name_list=[]
        feature_cn_name_list=[]
        #feature_aggmap_name_list=[]
        for feature_ent in feature_ents:
            transformation = feature_ent.get('transformation',None)
            feature = feature_ent.get("feature",None)
            if feature==None:
                raise ValueError("feature cannot be empty")
                
            feature_name = feature_ent.get("feature_name",None)            
            if feature_name == None:
                feature_name = feature
                
            feature_cn_name = feature_ent.get("feature_cn_name",None)            
            if feature_cn_name == None:
                feature_cn_name = feature_name
            print(feature)
            
            #transformation
            if transformation!=None:
                self.df,feature,feature_name,feature_cn_name=Transform(self.df,
                        transformation,feature,feature_name,feature_cn_name)
            print("feature",feature)   
            feature_list.extend(feature)
            
            if not isinstance(feature_name,list) :
                feature_name = [feature_name]
                feature_cn_name = [feature_cn_name]
            
            for _zip in zip(feature,feature_name,feature_cn_name):
                filter_name_list = list_combination([filter_name_list,feature_name_list],code="_")
                filter_cn_name_list = list_combination([filter_cn_name_list,feature_cn_name_list],code=".") 
                
                self.config_list.append(
                        domain(target,_zip[0],zip(filter_list,filter_name_list,filter_cn_name_list),
                            feature_ent.get("aggregation"),None)
                        )
                
            feature_name_list.extend(feature_name)            
            feature_cn_name_list.extend(feature_cn_name)
            for i in range(len(feature_name)):
                feature_aggmap_list.append(feature_ent.get("aggregation"))

        
        filter_len = max(len(filter_list),1)
        feature_len = len(feature_list)
        

        
        filter_name_list = list_combination([filter_name_list,feature_name_list],code="_")
        filter_cn_name_list = list_combination([filter_cn_name_list,feature_cn_name_list],code=".")    
        
        self.filter_name_list.extend(filter_name_list)
        self.filter_cn_name_list.extend(filter_cn_name_list)
        
        '''
        4 整合计算
        '''
        
        config_=[x for x in zip(list_rep(filter_list,feature_len),
                                      [target]*feature_len*filter_len,
                                      feature_list*filter_len,
                                      feature_aggmap_list*filter_len,
                                      filter_name_list,
                                      filter_cn_name_list)]
        self.config_.extend(config_)
        #return config_

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    