# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:10:22 2019

@author: liuyao
"""

import numpy as np
from feature_extraction.filter.filterbase import Filterbase
import re

class Map(Filterbase):
    
    def __init__(self,hit_list):
        self.hit_list = hit_list

    def get_function(self):
        def map_(df):                
            c = []
            for i,e in zip(self.fea_list,self.hit_list):
                b = (df[i].values==e).reshape(-1,1) if e else np.array([True] * df.shape[0]).reshape(-1,1)
                c.append(b)
            boo = np.concatenate(c,axis=1)
            mask = (boo.sum(axis=1) == len(self.fea_list)).reshape(-1,1)
            return mask
        return map_

class NotMap(Filterbase):
    
    def __init__(self,hit_list):
        self.hit_list = hit_list

    def get_function(self):
        def notmap_(df):                
            c = []
            for i,e in zip(self.fea_list,self.hit_list):
                b = (df[i].values!=e).reshape(-1,1) 
                c.append(b)
            boo = np.concatenate(c,axis=1)
            mask = (boo.sum(axis=1) == len(self.fea_list)).reshape(-1,1)
            return mask
        return notmap_


class MapIn(Filterbase):
    
    def __init__(self,hit_list,regex=True):
        if regex :
            self.hit_list = [re.compile(str(h)) for h in hit_list]
        else:
            self.hit_list = hit_list
        self.regex = regex

    def get_function(self):
        def mapin(df):                
            c = []
            for i,e in zip(self.fea_list,self.hit_list):
                if e is None:
                    np.array([True] * df.shape[0]).reshape(-1,1)
                elif self.regex: 
                    b = np.array([e.search(a) is not None for a in df[i].astype(str).values]).reshape(-1,1)
                else:
                    b = (df[i].values!=e).reshape(-1,1) 
                c.append(b)
            boo = np.concatenate(c,axis=1)
            mask = (boo.sum(axis=1) == len(self.fea_list)).reshape(-1,1)
            return mask
        return mapin

class MapNotIn(Filterbase):
    
    def __init__(self,hit_list):
        self.hit_list = hit_list

    def get_function(self):
        def mapin(df):                
            c = []
            for i,e in zip(self.fea_list,self.hit_list):
                if e is None:
                    np.array([True] * df.shape[0]).reshape(-1,1)
                else:
                    b = np.array([e not in a for a in df[i].values]).reshape(-1,1)
                c.append(b)
            boo = np.concatenate(c,axis=1)
            mask = (boo.sum(axis=1) == len(self.fea_list)).reshape(-1,1)
            return mask
        return mapin





