# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:55:33 2017

@author: languoxing
"""
# In[]
import matplotlib.pyplot as plt
import numpy as np
import math
# In[]
x1=np.arange(-5,1,0.01)
x2=np.arange(1,5,0.01)
# In[]
M=0.1
C=1
C1=M*x1+1
C2=M*x2+1
k=3
m=2
# In[]
def g(C,k,m,x):
    y=C/(1+np.exp(-k*(x-m)))
    return y
# In[]
delta_k=1
y1=g(C,k,m,x1)
y2=g(C,k+delta_k,m-delta_k/(k+delta_k),x2)

# In[]
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()

# In[]



