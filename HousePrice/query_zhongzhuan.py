# -*- coding:utf-8 -*- 
# from args import get_opts, nparrayToBytes, getSqlQuery, setLogging
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, Row, SQLContext

from pyspark.sql.types import FloatType
from datetime import datetime, timedelta

def read_hive(table):
  spark = SparkSession.builder.config(conf=SparkConf()).appName(table).enableHiveSupport().getOrCreate()
  df2 = spark.sql("select * from %s" % table)
  pandas_df = df2.toPandas()
  spark.stop()
  return pandas_df


def main(cur_date):
  # args, extra = get_opts()
  # Load SparkSession
  spark = SparkSession.builder.config(conf=SparkConf()).appName("zhongzhuan").enableHiveSupport().getOrCreate()
  b_date = datetime.strftime((datetime.strptime(cur_date, '%Y-%m-%d') + timedelta(days=-1)), '%Y-%m-%d')

  # cur_date = '2017-06-20'

  # query = getSqlQuery(args.sqlfile)
  #  df = spark.createDataFrame(spark.range(10).rdd.map(lambda x: x.id + 1.0/(x.id+0.1)), FloatType())
  #  df.show()
  query = """select 'zhongzhuang' as pred_type, a.to_tc as id,'' as name, substring(b.arrive_date, 0, 7) as ds, 
  sum(c.register_box_num) as numbox,sum(c.register_weight)/1000 as weight,sum(c.register_volume)/1000000 as bulk
  from fdm.fdm_trans_tc_proxy_book_chain a inner join fdm.fdm_trans_tc_proxy_book_item_chain b on a.sysno = b.proxy_book_sysno
  inner join fdm.fdm_trans_tc_transport_sheet_chain c on b.transport_sheet_sysno = c.sysno
  where
   a.status > 0 and a.deleted_flag = 0 and b.deleted_flag = 0
   and a.from_tc in  (50,51,52,53,54,55,56,57,355,379,380,388,389,390,391,392,393,394,395,396,397,423,591,631,754,755,756,757,758,759,760,761,765,810,811,832,838)
   and a.to_tc in  (50,51,52,53,54,55,56,57,355,379,380,388,389,390,391,392,393,394,395,396,397,423,591,631,754,755,756,757,758,759,760,761,765,810,811,832,838)
   and b.arrive_status = 20 and b.arrive_date is not null
   and a.start_date<'%s' and a.end_date>='%s'
   and b.start_date<'%s' and b.end_date>='%s'
   and c.start_date<'%s' and c.end_date>='%s'
   group by a.to_tc,substring(b.arrive_date, 0, 7) order by id, ds""" % tuple([cur_date] * 6)
  print(query)
  df2 = spark.sql(query)
  # df2.take(10).toDF().show()
  # df = spark.createDataFrame(df2)
  # print type(df2)
  # df2.show()
  pandas_df = df2.toPandas()
  pandas_df.to_csv("/data0/languoxing/zhongzhuan_"+cur_date+".csv")
  return pandas_df


if __name__ == "__main__":
  # setLogging('py4j', 30) # WARNING level
  # argument = argparser.pa
  df = main('2017-09-12')
  print(df)
