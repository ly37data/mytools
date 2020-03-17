from collections import Counter,namedtuple
import gzip,json,os,time


District = namedtuple('District','code,province,city,country,name,valid,attr')

def _bycode(code):
    return _index.get(code)

def _byname(name,**kwargs):
    return _bycode(_match_name(name,**kwargs))

def _tier(self):
    if 'tier'in self.attr:
        return self.attr['tier']
    if self.code%10000 == 0:
        return 'NA'
    city = _bycode(self.code //100)
    return city.attr.get('tier','T4-') if city else 'T4-'

def _region(self):
    if 'region'in self.attr:
        return self.attr['region']
    region = _bycode(self.code //10000)
    return region.attr.get('region','NA') if region else 'NA'

def _cur_city(self):
    if 'curCity' in self.attr:
        return District.bycode(self.attr['curCity'])
    if self.valid and self.code %10000 >9000:
        return self
    city = _bycode(self.code//100)
    if city and city.valid:
        return city
    return District.bycode(city.attr.get('curCity')) if city else None


District.bycode = _bycode
District.byname = _byname
District.tier = _tier
District.region = _region
District.curcity = _cur_city

regions = ['东北','华东','华中','华北','华南','西北','西南']
city_tiers = ['T1','T2a','T2b','T3','T4-']









