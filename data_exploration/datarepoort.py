# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:34:06 2019

@author: liuyao
"""
import sys
sys.path.append("F:\学习\kx_")

from mytools import data_exploration 
import pandas as pd 

debug=True

path = "F:\学习\拍拍贷\data\\"
nrows=1000 if debug else None

#分好多期，需要对应是还哪一期。
listing_info=pd.read_csv(path+"listing_info.csv",nrows=nrows)


cc = data_exploration.ProfileReport(listing_info)
cc.to_file(path+"cc1.html")

import pandas_profiling

cc = pandas_profiling.ProfileReport(listing_info)
cc.to_file(path+"cc.html")


