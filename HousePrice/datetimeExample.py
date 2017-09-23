# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:44:13 2017

@author: languoxing
"""
# In[]
import datetime as dt
import time as t
import pandas as pd
# In[]
now = dt.datetime.now()
print(now)
type(now)
# In[]
t.localtime()
# In[]
now.weekday()
# In[]
b=dt.datetime(2015, 1, 12)
# In[]
print(b)
# In[]
for i in range(1,10):
    print(i)
# In[]
guoqing = pd.DataFrame({
  'holiday': '1001',
  'ds': pd.to_datetime(['2014-10-01', '2015-10-01', '2016-10-01',
                        '2017-10-01', '2018-10-01']),
  'lower_window': 2,
  'upper_window': 3,
})
# In[]
with open('guoqing1.csv', 'a+') as f:
    guoqing.to_csv(f, header=False)
# In[]

# In[]

# In[]

# In[]

# In[]

