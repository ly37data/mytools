### feature_extraction

特征衍生工具

主要为了减少写重复代码的时间。所有特征衍生的逻辑通过配置文件来完成。

```python
import feature_extraction as fe
import pandas as pd

data = pd.read_csv('test.csv')

index=["Customers"]
feature_entries=[
        {"feature":["type"],
         "preprocessing":[pt.PassThroughStr],
         "aggregating":[pt.DummyCount(dummy_map={'1':'C1','2':'C2','3':'C3'}),
                        pt.UniqueCount,pt.Count],
         "feature_name_prefix":"DayOfWeek",
         "cn_desc":"类型",
                },
        {"feature":["Sales"],
         "preprocessing":[pt.PassThroughFloat],
         "aggregating":[pt.Sum,pt.Quantile(q=25),pt.Mean],
         "feature_name_prefix":"Sales",
         "cn_desc":"金额",
                },

        ]

config={
        "index":index,
        "feature_entries":feature_entries,      
        }

fe.dfs(data=data,config=config,prefix_name='behavior',prefix_cn_name='购买记录', 
                    cn_map=path+'data3map.csv',verbose=True,result=path+'tt.pkl',log_dir='./')
```

