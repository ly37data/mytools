# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:07:03 2019

@author: liuyao
"""

class Filterbase(object):
    
    def __init__(self):
        pass
    
    def set_arg(self,fea_list= None,filters_name= None,filters_cn_name= None):
        self.fea_list = fea_list
        self.filters_name = filters_name
        self.filters_cn_name = filters_cn_name
    
    def get_name(self):
        return self.filters_name+'_'

    def get_cn_name(self):
        return self.filters_cn_name+'_'
    
    
    