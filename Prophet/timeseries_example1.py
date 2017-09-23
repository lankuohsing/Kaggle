# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 18:11:43 2017

@author: languoxing
"""
# In[]
# Python
import pandas as pd
import numpy as np
from fbprophet import Prophet
# In[]
# Python
df = pd.read_csv('./example_wp_peyton_manning.csv')#读取数据
df['y'] = np.log(df['y'])#对要预测的序列求对数
df.head()

# In[]
# Python
m = Prophet()
m.fit(df);
# In[]
# Python
future = m.make_future_dataframe(periods=365)
future.tail()
# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

