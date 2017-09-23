# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:29:36 2017

@author: languoxing
"""
# In[]
import pandas as pd
import string
# In[]

df_real=pd.read_csv("forward_2017-06-01.csv")

# In[]
df_real_numbox=df_real[df_real.id<355&int(df_real.ds)>=2017]
# In[]
a=df_real.ds
# In[]
b=a[1]
# In[]
c=b.split('-')
# In[]
d=c[0]
# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

