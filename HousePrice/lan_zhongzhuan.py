# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:30 2017

@author: languoxing
"""

# In[]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
import datetime as dt
# In[]
#读取csv文件
all_zhongzhuan_50=pd.read_csv("./zhongzhuan_50.csv")
# In[]
#获取日期列
dateTime=all_zhongzhuan_50.iloc[:,3]
# In[]
#以“-”切割字符串形式的日期
dateTimeSplit=list(map(lambda x:x.split("-"), list(dateTime)))
# In[]
#将日期转化为int型
dateTimeInt=list(map(lambda x:list(map(int,x)),c))
# In[]
#s=[["1","2"],["3","4"]]
#r=list(map(lambda x:list(map(int,x)),s))
# In[]
#将int型日期转化为datetime格式
dateTime_standard=list(map(lambda x:dt.datetime(x[0],x[1],x[2]),dateTimeInt))
# In[]
dateTime_weekday=list(map(lambda x:x.weekday(),dateTime_standard))
# In[]
dateTime_weekday_df=pd.DataFrame(dateTime_weekday,columns=['weekday'])
# In[]
result1=pd.concat([all_zhongzhuan_50,dateTime_weekday_df],axis=1)
result1.to_csv("result1.csv")
# In[]

# In[]

# In[]

# In[]


