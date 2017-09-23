# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:39:49 2017

@author: languoxing
"""
# In[]
#导入机器学习linear_model库
from sklearn import linear_model
#导入交叉验证库
from sklearn import cross_validation
#导入数值计算库
import numpy as np
#导入科学计算库
import pandas as pd
#导入图表库
import matplotlib.pyplot as plt
import math

# In[]
df_predict=pd.read_csv("result0920.csv",encoding="gbk")#读取预测结果文件
df_predict_50=df_predict[df_predict.tc_id==51]#挑选出id为50的数据
numbox_predict_50_1=np.array(df_predict_50.carton_num[0:6])#挑选出2017年上半年的预测数据
# In[]
'''
numbox_real_50=np.array([#id=50的真实数据
        227078,
        654797,
        1094246,
        994043,
        1642203,
        936859,
        849392,
        1007122,
        423053

])
array_numbox_50_1=numbox_real_50[0:6]#真实数据中上半年的数据
'''
# In[]
numbox_real_50=np.array([#id=50的真实数据
        591989,
447440,
669682,
674249,
1079804,
605442,
519349,
656695,
293774

])
array_numbox_50_1=numbox_real_50[0:6]#真实数据中上半年的数据



# In[]
plt.plot(array_numbox_50_1,numbox_predict_50_1)
plt.show()
# In[]

# In[]
sale_per_month=1.5e8/6#上半年每个月的销售额
sale=np.ones((6,1))*sale_per_month
target_per_month=2.7e8/6
target=np.ones((6,1))*target_per_month#下半年每个月的目标销售额
# In[]
array_numbox_50_1=array_numbox_50_1.reshape(6,1)
numbox_predict_50_1=numbox_predict_50_1.reshape(6,1)
# In[]
Y=array_numbox_50_1
X=numbox_predict_50_1
# In[]
#Y=np.array([[10],[20],[30],[40],[50]])
#X=np.array([[1],[2],[3],[4],[5]])
# In[]
clf =linear_model.LinearRegression()
clf.fit (X,Y)
# In[]
#线性回归模型的斜率
clf.coef_
# In[]
clf.intercept_ 
# In[]
numbox_predict_50_2=np.array(df_predict_50.carton_num[6:12])
# In[]
numbox_predict_50_2_1=numbox_predict_50_2*clf.coef_+clf.intercept_ 
# In[]
numbox_predict_50_1_1=numbox_predict_50_1*clf.coef_+clf.intercept_ 
# In[]
numbox_predict_50_2_1=numbox_predict_50_2_1.T
# In[]
numbox_predict_50_2_2=numbox_predict_50_2_1*(2.7/1.5)/((sum(numbox_predict_50_2_1)/sum(numbox_predict_50_1_1)))
# In[]
numbox_real_50_2=numbox_real_50[6:9].reshape(3,1)
# In[]
error1=math.sqrt(sum(pow(numbox_predict_50_2[0:3].reshape(3,1)-numbox_real_50_2,2)))
# In[]
error2=math.sqrt(sum(pow(numbox_predict_50_2_1[0:3].reshape(3,1)-numbox_real_50_2,2)))
# In[]
error3=math.sqrt(sum(pow(numbox_predict_50_2_2[0:3].reshape(3,1)-numbox_real_50_2,2)))

# In[]
(2.7/1.5)/((sum(numbox_predict_50_2_1)/sum(numbox_predict_50_1_1)))
# In[]
2.7/1.5
# In[]

# In[]
plt.plot(numbox_real_50)
plt.show()
# In[]
plt.plot(df_predict_50.carton_num[0:9])
plt.show()
# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

