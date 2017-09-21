# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from numpy import linalg as la
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

def read_month(path, type):
  df = pd.read_excel(path, sheetname=3)
  df.columns = ['type', 'id', 'name', 'month', 'c', 'w', 'v']
  dd = read_month_df(df[df.type == type])
  return dd


def read_month_df(df):
  df.columns = ['type', 'id', 'name', 'month', 'c', 'w', 'v']
  df['name'].fillna("", inplace=True)
  df.sort_values(['type', 'id', 'month'], ascending=[1, 1, 1], inplace=True)
  df.reset_index(drop=True, inplace=True)
  df["type"] = df["type"].astype("category")
  # df["id"] = df["id"].astype("string")
  df['wperc'] = df['w'] / df['c']
  df['vperc'] = df['v'] / df['c']
  return df[df.month != 'null']
  # return df[(df.type == type) & (df.month != 'null')]

  # dc = df[df.type == type][df.month != 'null'].pivot_table('c', index="month", columns='id')
  # dw = df[df.type == type][df.month != 'null'].pivot_table('w', index="month", columns='id')
  # dv = df[df.type == type][df.month != 'null'].pivot_table('v', index="month", columns='id')

  # dd.fillna(0, inplace=True)
  # print(dd.columns)
  # for i in dd.columns:
  #   print(dd[[i]])
  # P, D, Q = la.svd(dd.iloc[:,:5], full_matrices=False)
  # # print(P, D, Q)
  # X_a = np.dot(np.dot(P, np.diag(D)), Q)
  # print(X_a)
  # print(np.std(dd), np.std(X_a), np.std(dd.iloc[:,:5] - X_a))

  # return dc, dw, dv

def pivot_month(path, type):
  df = pd.read_excel(path, sheetname=3)
  df.columns = ['type', 'id', 'name', 'month', 'c', 'w', 'v']
  return pivot_month_df(df[df.type == type])


def pivot_month_df(df):
  df["type"] = df["type"].astype("category")
  dc = df[df.month != 'null'].pivot_table('c', index="month", columns='id')
  return dc

if __name__ == "__main__":
  path = "./货量预测导出数据NEW.xlsm"

  print(path)
  df = read_month(path, u'TC正向')
  print(df)
