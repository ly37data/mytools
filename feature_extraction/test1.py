# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:47:02 2019

@author: liuyao
"""
import pandas as pd 
import time
from contextlib import contextmanager
import numpy as np

import sys
sys.path.append(r"F:\学习\kx_\featuretrools")
from mytools import synthesis


@contextmanager
def times():
    start_time = time.time()
   
    yield 
    end_time=time.time()
    print(f"耗时:{round(end_time-start_time,2)}s")
    
from functools import reduce

def list_combination(lists,code=""):
    def func1(x,y):
        return [str(i)+code+str(j) for i in x for j in y]
    return reduce(func1,lists)

start_time = time.time()

path = "F:\学习\拍拍贷\data\\"
nrows=100000

user_repay_logs = pd.read_csv(path+"user_repay_logs.csv",nrows=nrows)
listing_info=pd.read_csv(path+"listing_info.csv",parse_dates=["auditing_date"],
                         infer_datetime_format="%Y-%m-%d",nrows=nrows)

train = pd.read_csv(path+"train.csv", parse_dates=['auditing_date', 'due_date', 'repay_date'],
                    infer_datetime_format="%Y-%m-%d",nrows=nrows,)

train.rename(columns={"auditing_date":"apply_time"},inplace=True)

tt=listing_info.merge(train[["user_id","apply_time"]],on="user_id",how="inner")
tt["deal_time"] = tt["apply_time"] - tt["auditing_date"]
tt["deal_time"] = tt["deal_time"].apply(lambda x:x.days)

target=["user_id"]
time_index={
        "time_index":"apply_time",
        "time_cal":"auditing_date",
        "time_window":[30,90,180,360,None]
        }
filters=[{
        #"filters_data":,
        "filters_feature":["term","rate"],
        "filters":[
                {"filters_value":[None,None],
                 "filters_name":"ALL",
                 "filters_cn_name":"111"
                 
                 },
                 {"filters_value":[9,7.6],
                 "filters_name":"term9rate76",
                 "filters_cn_name":"期数9费率76"
                 
                 },
                ]
        }]
feature_ent=[
        {"feature":["principal"],
         #"transformation":
         "aggregation":["count","sum"],
         "feature_name":"principal",
         "feature_cn_name":"申请金额",
                }
        
        ]
config=[
        {
        "target":target,
        "time_index":time_index,
        "filters":filters,
        "feature_ent":feature_ent        
        }
        ]


sy=synthesis(config=config)   
res,name_pd=sy.dfs(tt)

end_time=time.time()
print(f"耗时:{round(end_time-start_time,2)}s")





class synthesis():
    def __init__(self,config):
        self.config = config
        self.config_s()
        self.columns_name_cn = pd.DataFrame(columns=["feature_name","cn_name"])
        
    def dfs(self,df):
        
        df["deal_time"] = df["apply_time"] - df["auditing_date"]
        df["deal_time"] = df["deal_time"].apply(lambda x:x.days)
        result=[]
        name_cn_list=[]
        for target_ in self.config_:
            gb_re=df.query(target_[0]).groupby(target_[1])[target_[2]].agg(target_[3])
            gb_re.columns = [target_[4]+"_"+x for x in target_[3]]
            
            name_cn = [target_[5]+"_"+agg_name_map.get(x) for x in target_[3]]
            name_cn_list.append(pd.DataFrame(np.array([[target_[4]+"_"+x for x in target_[3]],name_cn]).T,
                                         columns=["feature_name","cn_name"]))
            result.append(gb_re)
        feature_result = pd.concat(result,axis=1).reset_index()
        return  feature_result,pd.concat(name_cn_list,ignore_index=True)
        
    def config_s(self):
        ls_config = len(config)
        if ls_config==0:
            print()
        for val in config:
            target=val.get("target")
            time_index=val.get("time_index")
            filters=val.get("filters")
            feature_ents = val.get("feature_ent")
            aa="deal_time"
            cc=[aa,["apply_time","auditing_date"]]
            time_filter_list=list(map(lambda x:"0<"+aa+"<="+str(x) if x!=None else aa+"<=inf",time_index.get("time_window")))
            time_filter_name_list=list(map(lambda x:str(x)+"d" if x!=None else "",time_index.get("time_window")))
            time_filter_cn_name_list=list(map(lambda x:"近"+str(x)+"天" if x!=None else "",time_index.get("time_window")))    
            value_filter_list=[]
            value_filter_name_list=[]
            value_filter_cn_name_list=[]
            for filter_ in filters:
                filters_features=filter_.get("filters_feature")
                for fil in filter_.get("filters"):
                    filters_value=fil.get("filters_value")
                    minlist=[]
                    for features_,values_ in zip(filters_features,filters_value):
                        if values_!=None:
                            minlist.append(features_+"=="+str(values_))
                        
                    minlist.append(features_+"!=' '")
                    value_filter_list.append(" and ".join(minlist))
                    value_filter_name_list.append(fil.get("filters_name"))
                    value_filter_cn_name_list.append(fil.get("filters_cn_name"))
                    
            filter_list = list_combination([value_filter_list,time_filter_list],code=" and ")
            
            feature_list=[]
            feature_aggmap_list=[]
            feature_name_list=[]
            feature_cn_name_list=[]
            #feature_aggmap_name_list=[]
            for feature_ent in feature_ents:
                feature_list.append(feature_ent.get("feature"))
                feature_aggmap_list.append(feature_ent.get("aggregation"))
                feature_name_list.append(feature_ent.get("feature_name"))
                feature_cn_name_list.append(feature_ent.get("feature_cn_name"))
                
            
            feature_len = len(filter_list)
            filter_name_list = list_combination([value_filter_name_list,time_filter_name_list,feature_name_list],code="_")
            filter_cn_name_list = list_combination([value_filter_cn_name_list,
                                                    time_filter_cn_name_list,feature_cn_name_list],code="_")
            
            self.config_=[x for x in 
                          zip(filter_list,target*feature_len,feature_list*feature_len,
                              feature_aggmap_list*feature_len,filter_name_list,filter_cn_name_list)]

    


    
    
    
    
    
    



