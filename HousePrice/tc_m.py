# -*- coding:utf-8 -*-

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')


import argparse
import heapq
import logging
import random
import time
import traceback
import uuid

import numpy as np
import pandas as pd
from fbprophet import Prophet
from scipy.stats import randint as sp_randint
from scipy.stats.distributions import uniform
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler

from data_utils import read_month, read_month_df, pivot_month
from tc import adj_r2, holidays, mape

BOXCOX_LAMBDA = 0.10

# create logger
logger_name = "example"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# create file handler
log_path = "./log.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)

# create formatter
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d\n%(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

# add handler and formatter to logger
fh.setFormatter(formatter)
logger.addHandler(fh)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 100)


def read_data():
  return pd.read_csv("tc51.txt")


# jd_618 = pd.DataFrame({
#   'holiday': '618',
#   'ds': pd.to_datetime(['2014-06-18', '2015-06-18', '2016-06-18',
#                         '2017-06-18', '2018-06-18']),
#   'lower_window': -50,
#   'upper_window': 10,
# })

month_3 = pd.DataFrame({
  'holiday': 'month_3',
  'ds': pd.to_datetime(['2014-03-01', '2015-03-01', '2016-03-01',
                        '2017-03-01', '2018-03-01']),
  'lower_window': 0,
  'upper_window': 1,
})
month_4 = pd.DataFrame({
  'holiday': 'month_4',
  'ds': pd.to_datetime(['2014-04-01', '2015-04-01', '2016-04-01',
                        '2017-04-01', '2018-04-01']),
  'lower_window': 0,
  'upper_window': 1,
})

jd_5 = pd.DataFrame({
  'holiday': '5',
  'ds': pd.to_datetime(['2014-05-01', '2015-05-01', '2016-05-01',
                        '2017-05-01', '2018-05-01']),
  'lower_window': 0,
  'upper_window': 1,
})
jd_6 = pd.DataFrame({
  'holiday': '618',
  'ds': pd.to_datetime(['2014-06-01', '2015-06-01', '2016-06-01',
                        '2017-06-01', '2018-06-01']),
  'lower_window': 0,
  'upper_window': 1,
})

month_7 = pd.DataFrame({
  'holiday': 'month_7',
  'ds': pd.to_datetime(['2014-07-01', '2015-07-01', '2016-07-01',
                        '2017-07-01', '2018-07-01']),
  'lower_window': 0,
  'upper_window': 1,
})

month_8 = pd.DataFrame({
  'holiday': 'month_8',
  'ds': pd.to_datetime(['2014-08-01', '2015-08-01', '2016-08-01',
                        '2017-08-01', '2018-08-01']),
  'lower_window': 0,
  'upper_window': 1,
})

month_9 = pd.DataFrame({
  'holiday': 'month_9',
  'ds': pd.to_datetime(['2014-09-01', '2015-09-01', '2016-09-01',
                        '2017-09-01', '2018-09-01']),
  'lower_window': 0,
  'upper_window': 1,
})

guoqing = pd.DataFrame({
  'holiday': '1001',
  'ds': pd.to_datetime(['2014-10-01', '2015-10-01', '2016-10-01',
                        '2017-10-01', '2018-10-01']),
  'lower_window': 0,
  'upper_window': 1,
})
tm_1111 = pd.DataFrame({
  'holiday': '1111',
  'ds': pd.to_datetime(['2014-11-01', '2015-11-01', '2016-11-01',
                        '2017-11-01', '2018-11-01']),
  'lower_window': 0,
  'upper_window': 1,
})
festival = pd.DataFrame({
  'holiday': 'festival',
  'ds': pd.to_datetime(['2014-02-01', '2015-02-01', '2016-02-01',
                        '2017-01-01', '2018-02-01']),
  'lower_window': 0,
  'upper_window': 1,
})
festival_b = pd.DataFrame({
  'holiday': 'festival_b',
  'ds': pd.to_datetime(['2014-01-01', '2015-01-01', '2016-01-01',
                        '2016-12-01', '2018-01-01']),
  'lower_window': 0,
  'upper_window': 1,
})

holidays = pd.concat((festival, jd_5, jd_6, guoqing, tm_1111, festival_b))

month_changepoints = ['2015-02-01', '2016-02-01']


def invboxcox(y, ld):
  if ld == 0:
    return (np.exp(y) - 1)
  else:
    return (np.exp(np.log(ld * y + 1) / ld) - 1)


def train(data):
  m = Prophet(yearly_seasonality=True, seasonality_prior_scale=50, changepoint_prior_scale=0.35).fit(data)
  # import pydoop.hdfs as hdfs
  # f = hdfs.open('/user/myuser/filename', mode="w")
  # f = open("models/m.pkl", "w")
  # pickle.dump(m, f)
  return m


def predict(data, m):
  # dd['y'] = np.log(dd['y'])
  # m = pickle.load(open("models/m.pkl", "r"))
  dsplit = len(data)
  future = m.make_future_dataframe(dsplit, freq='MS')
  fcst = m.predict(future)
  # print(fcst)
  # m.plot(fcst)
  # m.plot_components(fcst)
  # plt.show()
  return fcst


def test(fcst, dd):
  # a = fcst[-dsplit:].set_index('ds').shift(1, freq='D')
  # print(a)
  # comp = pd.DataFrame()
  # comp['ds'] = dtest['ds']
  # comp = comp.set_index('ds')
  # comp['original'] = dtest.set_index('ds')['y']
  # # comp['ds2'] = a['ds']
  # comp['predict'] = a['yhat']
  # print(comp)
  # print(len(fcst['yhat']))



  # fcst['yhat'] = fcst['yhat']
  # fcst['y_real'] = dd['y']
  # print(np.mean(fcst[fcst.ds.isin(festival.ds)]['y_real'])
  # fcst.loc[fcst.ds=='2017-01-01','yhat'] = np.mean(fcst[fcst.ds.isin(festival.ds)]['y_real'])
  # print(fcst)
  real_r2 = adj_r2(fcst['y_real'][:-dsplit], fcst['yhat'][:-dsplit])
  test_r2 = adj_r2(fcst['y_real'][-dsplit:], fcst['yhat'][-dsplit:])
  test_r2_process = adj_r2(fcst['y_process'][-dsplit:], fcst['yhat'][-dsplit:])
  test_mape = mape(fcst['y_real'][-dsplit:], fcst['yhat'][-dsplit:])
  test_mape_process = mape(fcst['y_process'][-dsplit:], fcst['yhat'][-dsplit:])
  return real_r2, test_r2, test_mape, test_r2_process, test_mape_process


def g_s():
  param_grid = [{
    'seasonality_prior_scale': [round(x * 0.1, 1) for x in range(1, 10)],  # np.linspace(0.1, 1, 10), range(1, 10, 2), #
    # 'yearly_seasonality': [True],
    # 'mcmc_samples': [100],
    'n_changepoints': [5, 10],
    'changepoint_prior_scale': [round(x * 0.01, 2) for x in range(5, 30, 5)],  # np.linspace(0.05, 0.25, 5),[0.05], #
    'holidays_prior_scale': range(1, 10, 2),  # range(10,100,10), #
    # 'holidays': [holidays],
    # 'interval_width': [0.8, 0.95]
  }]
  for p in ParameterGrid(param_grid):
    yield p


def random_search():
  param_grid = {
    'seasonality_prior_scale': uniform(0.1, 1),  # np.linspace(0.1, 1, 10), range(1, 10, 2), #
    # 'yearly_seasonality': [True],
    # 'mcmc_samples': [100],
    'n_changepoints': sp_randint(2, 10),
    'changepoint_prior_scale': uniform(0.05, 0.15),  # np.linspace(0.05, 0.25, 5),[0.05], #
    'holidays_prior_scale': sp_randint(1, 10),  # range(10,100,10), #
    # 'holidays': [holidays],
    # 'interval_width': [0.8, 0.95]
  }
  param_list = list(ParameterSampler(param_grid, n_iter=100))
  return [dict((k, round(v, 1)) for (k, v) in d.items()) for d in param_list]


def random_search2():
  param_grid = {
    'seasonality_prior_scale': uniform(1, 10),  # np.linspace(0.1, 1, 10), range(1, 10, 2), #
    # 'yearly_seasonality': [True],
    # 'mcmc_samples': [100],
    'n_changepoints': sp_randint(1, 20),
    'changepoint_prior_scale': uniform(0.05, 0.85),  # np.linspace(0.05, 0.25, 5),[0.05], #
    'holidays_prior_scale': sp_randint(1, 50),  # range(10,100,10), #
    # 'holidays': [holidays],
    # 'interval_width': [0.8, 0.95]
  }
  param_list = list(ParameterSampler(param_grid, n_iter=100))
  return [dict((k, round(v, 1)) for (k, v) in d.items()) for d in param_list]


def predict_job(p):
  # holidays = pd.concat(tuple(random.sample([festival, jd_5, jd_6, guoqing, tm_1111, festival_b], 4)))
  pp = Prophet(holidays=holidays, yearly_seasonality=True, **p)
  # pp.add_seasonality(name='monthly', period=30.5, fourier_order=5)
  try:
    print("start fit", p)
    m = pp.fit(dtrain)
  except Exception as e:
    traceback.print_exc()
    # try:
    #   pp = Prophet(holidays=holidays, yearly_seasonality=True, mcmc_samples=500, **p)
    #   m = pp.fit(dtrain)
    # except:
    #   traceback.print_exc()
    #   return
    return
    # traceback.print_exc()
    # print("exception,", p)
  print(len(result))
  fcst = predict(dtest, m)
  # if i == '50':
  #   fcst.loc[fcst.ds == '2017-01-01', 'yhat'] = np.mean(fcst[fcst.ds.isin(festival.ds)]['y_real'])
  print("predict end")

  fcst.set_index("ds", inplace=True)
  fcst['y_real'] = dds['y_real']
  fcst['y_process'] = dds['y']
  # fcst['yhat_log'] = fcst['yhat']
  # fcst['yreal_log'] = dd['y']
  # fcst['yhat'] = np.exp(fcst['yhat'])
  # fcst['y_real'] = np.exp(dd['y'])
  #
  # for i in fcst.columns[2:]:
  #   fcst[i] = invboxcox(fcst[i].clip(lower=1), BOXCOX_LAMBDA)
  # fcst['yhat_boxcox'] = fcst['yhat']
  # fcst['yreal_boxcox'] = dd['y']
  # fcst['yhat'] = invboxcox(fcst['yhat'].clip(lower=1), BOXCOX_LAMBDA)
  # fcst['y_real'] = invboxcox(dd['y'].clip(lower=1), BOXCOX_LAMBDA)

  # m.plot_components(fcst)
  # m.plot(fcst)
  # plt.show()
  print("start test")
  real_r2, test_r2, test_mape, test_r2_process, test_mape_process = test(fcst, dd)
  return {"p": p, "real_r2": real_r2, "test_r2": test_r2, "test_mape": test_mape,
          "test_r2_process": test_r2_process, "test_mape_process": test_mape_process, 'fcst': fcst}
  # print("end test")


def save_db(df, predict_month_start):
  from db import Mysql
  tc_db = Mysql('tc')
  for d in df.to_dict(orient='records'):
    d['id'] = str(uuid.uuid4())
    d['area'] = ''
    d['city'] = ''
    d['predicted_time'] = predict_month_start
    c = tc_db.save('predict_month', d, ['tc_id', 'predict_month', 'predicted_time', 'predict_type'])
    print(c)
  tc_db.commit()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--type', '-t',
    dest='type',
    type=str,
    default='forward',
    help='predict type'
  )
  parser.add_argument(
    '--parallel', '-p',
    dest='parallel',
    type=int,
    default=8,
    help='parallel cpu num to search param'
  )
  parser.add_argument(
    '--tc_id', '-id',
    dest='tc_id',
    type=int,
    default=50,
    help='default tc_id'
  )
  parser.add_argument(
    '--svd', '-svd',
    dest='svd',
    type=bool,
    default=False,
    help='use svd to deduct dimensions'
  )
  parser.add_argument(
    '--predict_month_start', '-m_s',
    dest='predict_month_start',
    type=str,
    default='2017-01-01',
    help='start month of predict'
  )
  parser.add_argument(
    '--month_amount', '-m_c',
    dest='month_amount',
    type=int,
    default=12,
    help='the amount of the predict month'
  )
  parser.add_argument('--use_spark', dest='use_spark', type=bool, default=False, help='data from aip.jd.com spark')
  parser.add_argument('--input-path', dest='input', type=str, default='', help='the input from last of pipeline')
  parser.add_argument('--hive', dest='hive', type=bool, default=False, help='data from hive')

  FLAGS, unparsed = parser.parse_known_args()
  # dsplit = 12

  # dd = read_data()[-dsplit-30:].reset_index(drop=True)
  # # dd['y'] = np.log(dd['y'])
  # # print(pd.DataFrame(data=dd, index=range(dsplit+12)))
  # dtrain = dd[:-dsplit]
  # dtest = dd[-dsplit:]
  # m = train(dtrain)
  # fcst = predict(dtest)
  # test(fcst, dd)
  print(FLAGS.tc_id)
  path = "./货量预测导出数据NEW.xlsm"
  type_dict = {"forward": u"TC正向", "neipei": u"内配-TC", "zhongzhuan": u"中转"}

  if FLAGS.use_spark:
    from query_zhongzhuan import read_hive

    df_hive = read_hive(FLAGS.input)
    df_calc = read_month_df(df_hive)
  print(FLAGS.predict_month_start)
  print(FLAGS.hive)
  if not FLAGS.hive:    #默认为false
    # df_calc = read_month(path, type_dict.get(FLAGS.type))
    df_csv = pd.read_csv("/data0/languoxing/" + FLAGS.type + "_" + "2017-09-12" + ".csv", index_col=0)
    df_calc = read_month_df(df_csv)
    df_calc = df_calc[df_calc.month<FLAGS.predict_month_start[:7]]
  elif FLAGS.hive:
    if FLAGS.type == "forward":
      from query_forward import main

      df_calc = main(FLAGS.predict_month_start)
    elif FLAGS.type == "neipei":
      from query_neipei import main

      df_calc = main(FLAGS.predict_month_start)
    elif FLAGS.type == "zhongzhuan":
      from query_zhongzhuan import main

      df_calc = main(FLAGS.predict_month_start)
  # logger.info(df_calc)
  # ids = df_calc[df_calc.type == type_dict.get(FLAGS.type)]['id'].unique()
  ids = df_calc['id'].unique()

  if FLAGS.svd:
    from svd_data import linsvd

    svd_df = pivot_month(path, type_dict.get(FLAGS.type))
    svd_df.fillna(0, inplace=True)
    print(svd_df)
    # X = np.array([[1, 2, 3], [2, 5, 106], [0, 5, 5]])
    X = svd_df.values
    # y = sksvd(X)
    # y = pd.DataFrame(y, index=df.index, columns=df.columns)
    # print(y)
    X_a = linsvd(X, n_components=1)
    svd_y = pd.DataFrame(X_a, index=svd_df.index, columns=svd_df.columns)

  for i in ids:
    #if FLAGS.tc_id and i != FLAGS.tc_id:
      #continue
    dd = df_calc[df_calc.id == i]
    # if i == 56:
    #   dd = df[[i]][9:]

    # from datetime import datetime
    # dd.loc[dd.month==datetime.strptime('2016-02-01', '%Y-%m-%d'), 'c'] = 0

    dd.dropna(inplace=True)
    dd.reset_index(inplace=True, drop=True)
    # dd.columns = ['ds', 'y']
    dd.rename(index=str, columns={'month': 'ds', 'c': 'y_real'}, inplace=True)
    dds = dd.set_index("ds")
    dds.index = pd.to_datetime(dds.index, format="%Y-%m-%d")

    # 运用svd后的预测
    if FLAGS.svd:
      dds['y'] = svd_y[[i]]

    # 利用原有数据预测
    else:
      dds['y'] = dds['y_real']
    dd = dds.reset_index()

    # dds.plot()
    # plt.show()

    mean_w, std_w, min_w, max_w = np.mean(dd['wperc']), np.std(dd['wperc']), np.min(dd['wperc']), np.max(dd['wperc'])
    # a_w, b_w = (min_w - mean_w) / std_w, (max_w - mean_w) / std_w
    # r_w = truncnorm.rvs(a_w, b_w, size=12)

    mean_v, std_v, min_v, max_v = np.mean(dd['vperc']), np.std(dd['vperc']), np.min(dd['vperc']), np.max(dd['vperc'])
    # a_v, b_v = (min_v - mean_v) / std_v, (max_v - mean_v) / std_v
    # r_v = truncnorm.rvs(a_v, b_v, size=12)



    # dw_fit = dw[[i]]
    # dw_fit.dropna(inplace=True)
    # dw_fit.reset_index(inplace=True)
    # dw_fit.columns = ['ds', 'y']
    #
    # dv_fit = dv[[i]]
    # dv_fit.dropna(inplace=True)
    # dv_fit.reset_index(inplace=True)
    # dv_fit.columns = ['ds', 'y']

    if (len(dd) < 6):
      continue

    dsplit = int(len(dd) * 0.2)
    # dd["y"] = boxcox(dd['y'].clip(lower=0) + 1, lmbda=BOXCOX_LAMBDA)
    # dd['y'] = invboxcox(dd['y'].clip(lower=1), BOXCOX_LAMBDA)

    # dd['real_y'] = dd['y']
    # dd['y'] = np.log(dd['y'])

    # print(pd.DataFrame(data=dd, index=range(dsplit+12)))
    dtrain = dd[:-dsplit]
    logger.info(dtrain)
    dtest = dd[-dsplit:]
    logger.info(dtest)
    # m = train(dtrain)
    result = []

    start = time.time()
    result = Parallel(n_jobs=FLAGS.parallel)(delayed(predict_job)(p) for p in random_search())

    # for p in g_s():
    #   # p = {'holidays_prior_scale': 10, 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 9, 'n_changepoints': 15, 'yearly_seasonality': True}
    #   # p = {'holidays_prior_scale': 4.0, 'changepoint_prior_scale': 0.1187, 'seasonality_prior_scale': 0.3526, 'n_changepoints': 9.0}
    #   r = predict_job(p)
    #   result.append(r)
    # r = predict_job({'holidays_prior_scale': 4.0, 'changepoint_prior_scale': 0.1187, 'seasonality_prior_scale': 0.3526, 'n_changepoints': 9.0})
    # result.append(r)
    # print(r)
    duration = time.time() - start
    print("end training is %.2f s" % duration)
    result = [r for r in result if r is not None and 'test_mape' in r]
    cheap = heapq.nsmallest(10, result, key=lambda s: s['test_mape'])
    print("----------------")
    for c in cheap:
      # print(c.fcst)
      # del c['fcst']
      logger.info(c)
    try:
      if cheap[0].get("test_mape") > 0.3:
        logger.error("tc_id is %s, len(dd) is %d, result is %s" % (str(i), len(dd), cheap[0]))
      else:
        logger.info("tc_id is %s, len(dd) is %d, result is %s" % (str(i), len(dd), cheap[0]))
      model_c = Prophet(holidays=holidays, yearly_seasonality=True, mcmc_samples=500, **cheap[0].get('p')).fit(dd)
      # model_w = Prophet(holidays=holidays, yearly_seasonality=True, mcmc_samples=500, **cheap[0].get('p')).fit(dw_fit)
      # model_v = Prophet(holidays=holidays, yearly_seasonality=True, mcmc_samples=500, **cheap[0].get('p')).fit(dv_fit)
    except:
      logger.info("error")
      model_c = Prophet(holidays=holidays, yearly_seasonality=True, mcmc_samples=500, **cheap[1].get('p')).fit(dd)
      # model_w = Prophet(holidays=holidays, yearly_seasonality=True, mcmc_samples=500, **cheap[1].get('p')).fit(dw_fit)
      # model_v = Prophet(holidays=holidays, yearly_seasonality=True, mcmc_samples=500, **cheap[1].get('p')).fit(dv_fit)
    future12_c = model_c.make_future_dataframe(FLAGS.month_amount, freq='MS')
    fcst12_c = model_c.predict(future12_c)
    # model_c.plot_components(fcst12_c)
    # model_c.plot(fcst12_c)
    # plt.savefig("test.jpg")
    # plt.show()
    fcst12_c.set_index("ds", inplace=True)
    fcst12_c['y_real'] = dds['y_real']

    # future12_w = model_w.make_future_dataframe(12, freq='MS')
    # fcst12_w = model_w.predict(future12_w)
    # # model_c.plot_components(fcst12_c)
    # # model_c.plot(fcst12_c)
    # # plt.savefig("test.jpg")
    # # plt.show()
    # fcst12_w.set_index("ds", inplace=True)
    # dw_fit.set_index("ds", inplace=True)
    # dw_fit.index = pd.to_datetime(dw_fit.index, format="%Y-%m-%d")
    # fcst12_w['y_real'] = dw_fit['y']
    #
    # future12_v = model_v.make_future_dataframe(12, freq='MS')
    # fcst12_v = model_v.predict(future12_v)
    # # model_c.plot_components(fcst12_c)
    # # model_c.plot(fcst12_c)
    # # plt.savefig("test.jpg")
    # # plt.show()
    # fcst12_v.set_index("ds", inplace=True)
    # dv_fit.set_index("ds", inplace=True)
    # dv_fit.index = pd.to_datetime(dv_fit.index, format="%Y-%m-%d")
    # fcst12_v['y_real'] = dv_fit['y']

    df = pd.DataFrame()
    df['predict_month'] = fcst12_c.index
    df.set_index("predict_month", inplace=True)
    df['tc_id'] = str(i)
    df['tc_name'] = dd["name"][0]
    df['carton_num'] = fcst12_c['yhat'].astype("int")
    df['predict_type'] = type_dict.get(FLAGS.type)

    df = df[df.index >= FLAGS.predict_month_start]
    df.loc[df.carton_num < 0, 'carton_num'] = 0
    r_w = [min(max_w, max(min_w, random.gauss(mean_w, std_w))) for _ in range(len(df))]
    df['weight'] = df['carton_num'] * r_w  # .astype("int")
    r_v = [min(max_v, max(min_v, random.gauss(mean_v, std_v))) for _ in range(len(df))]
    df['volume'] = df['carton_num'] * r_v  # .astype("int")
    df.reset_index(inplace=True)

    # logger.info(df)
    # logger.info(fcst12_c)

    #save_db(df, FLAGS.predict_month_start)
    if i==50:
        header1=True
    else:
        header1=False
    with open('result0920.csv', 'a') as f:
        df.to_csv(f,header=header1,encoding='gbk')
    

    # logger.info(fcst12_w)
    # logger.info(fcst12_v)
