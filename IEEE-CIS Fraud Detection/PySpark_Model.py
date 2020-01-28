#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.12.20
# os.environ['PYSPARK_PYTHON'] = '/home/aigege/.conda/envs/py3-conda/bin/python3.6'
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.storagelevel import StorageLevel
from pyspark import storagelevel
from pyspark.sql.types import TimestampType,FloatType,IntegerType
import numpy as np
import gc
import time
from contextlib import contextmanager
import warnings
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F
from pyspark import SparkConf
from sklearn.metrics import roc_auc_score
import pandas as pd
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.stat import Summarizer
from sklearn.model_selection import StratifiedKFold
import datetime
import copy
import os
# memory = '8g'
# pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
# os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
# COLUMNS WITH STRINGS
str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
            'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

# FIRST 53 COLUMNS
cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
       'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
       'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
       'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
       'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
       'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
       'M5', 'M6', 'M7', 'M8', 'M9']

# V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
# https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
v =  [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]

#v += [96, 98, 99, 104] #relates to groups, no NAN
v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN
v += [124, 127, 129, 130, 136] # relates to groups, no NAN

# LOTS OF NAN BELOW
v += [138, 139, 142, 147, 156, 162] #b1
v += [165, 160, 166] #b1
v += [178, 176, 173, 182] #b2
v += [187, 203, 205, 207, 215] #b2
v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2
v += [218, 223, 224, 226, 228, 229, 235] #b3
v += [240, 258, 257, 253, 252, 260, 261] #b3
v += [264, 266, 267, 274, 277] #b3
v += [220, 221, 234, 238, 250, 271] #b3

v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN
v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN
v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN
#v += [332, 325, 335, 338] # b4 lots NAN

cols += ['V'+str(x) for x in v]
dtypes = {}
for c in cols+['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,34)]:
    dtypes[c] = 'float'
dtypes['TransactionID'] = 'int'
for c in str_type: dtypes[c] = 'string'
warnings.simplefilter(action='ignore', category=FutureWarning)
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
# FREQUENCY ENCODE TOGETHER
def encode_FE(df, cols):
    print('encode_FE')
    for col in cols:
        print('encode_FE:{}'.format(col))
        # 生成对应的每个值的出现频率字典
        tmp1 = df.groupBy(col).count().orderBy('count',ascending = False)
        tmp_sum = tmp1.select(F.sum('count')).collect()[0][0]

        tmp1 = tmp1.withColumn('count',(tmp1['count']/tmp_sum).cast('float'))

        # tmp1 = tmp1.withColumn(col, tmp1[col].cast('string'))
        # tmp1 = tmp1.withColumn('count', tmp1['count'].cast('string'))
        # df = df.withColumn(col,df[col].cast('string'))
        # 问题8：因为使用了toPandas来获取数据，所以需要安装pyarrow这个包，我一开始安装pyspark的时候没有默认安装，导致每次这一步的时候卡半小时，装了以后就很快了。
        #具体参考spark官网关于arrow加速的章节。
        # 最终发现，collect的速度比topandas快了几十倍，而且topandas容易内存爆炸，所以还是选择了collect来获取数据。
        tmp = tmp1.collect()

        # tmp2 = tmp1.toPandas()
        tmp = dict(zip([str(i[0]) for i in tmp], [str(i[1]) for i in tmp]))

        # tmp = tmp1.collect()
        # tmp = dict(zip([str(i[0]) for i in tmp],[str(i[1]) for i in tmp]))
        tmp['-1'] = '-1'
        nm = col + '_FE'
        df = df.withColumn(nm, df[col].cast('string'))

        # 问题9：注释掉的replace的写法这里相当于pandas里面，df.map(dict)的效果。不过replace也会导致内存占用过大，最终还是选择自定义udf函数的方式来实现map功能，
        # 和pandas.apply功能类似。后面的函数里面实现一样，不再赘述。
        # df = df.replace(to_replace=tmp, subset=[nm])
        map_udf = F.udf(lambda x:tmp.get(x, x))
        df = df.withColumn(nm, map_udf(nm))
        df = df.withColumn(nm, df[nm].cast('float'))

        tmp1.unpersist()
        del tmp,tmp1
        gc.collect()
    return df
# LABEL ENCODE
def encode_LE(col, df):
    print('encode_LE')
    stringIndexer = StringIndexer(inputCol=col, outputCol=col + "_encode", handleInvalid="keep",
                                  stringOrderType="frequencyDesc")
    model = stringIndexer.fit(df)
    df = model.transform(df)
    df = df.drop(col).withColumnRenamed(col + "_encode",col)
    return df
# COMBINE FEATURES
def encode_CB(df,col1, col2 ):
    print('encode_CB')
    nm = col1 + '_' + col2
    df= df.withColumn(nm,F.format_string('%s_%s',df[col1].cast('string'), df[col2].cast('string')))
    df = encode_LE(nm,df)
    return df
# GROUP AGGREGATION MEAN AND STD
def encode_AG(main_columns, uids,df, aggregations=['mean'],
              fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    print('encode_AG')
    for main_column in main_columns:
        print('encode_AG:{}'.format(main_column))
        # if main_column=='M4':
        #     continue
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column + '_' + col + '_' + agg_type
                # temp_df = df[[col, main_column]]
                if usena:
                    df = df.withColumn(main_column + 'new', F.when(df[main_column]!= -1, df[main_column]).otherwise(np.nan)).drop(df[main_column]) \
                        .withColumnRenamed(main_column + 'new',main_column)

                tmp1 = df.groupBy(col).agg({main_column:agg_type})
                tmp_name = tmp1.columns[1]
                tmp1 = tmp1.withColumnRenamed(tmp_name,new_col_name)

                tmp = tmp1.collect()
                tmp = dict(zip([str(i[0]) for i in tmp], [str(i[1]) for i in tmp]))
                df = df.withColumn(new_col_name, df[col].cast('string'))

                map_udf = F.udf(lambda x: tmp.get(x, x))
                df = df.withColumn(new_col_name, map_udf(new_col_name))
                df = df.withColumn(new_col_name, df[new_col_name].cast('float'))
                if fillna:
                    df = df.fillna({new_col_name:-1})
                tmp1.unpersist()
                del tmp1,tmp
                gc.collect()
    return  df
# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, df):
    print('encode_AG2')
    for main_column in main_columns:
        print('encode_AG2:{}'.format(main_column))
        for col in uids:
            tmp1 = df.groupby(col).agg(F.expr('count(distinct {})'.format(main_column)).alias('nunique'))
            # tmp = tmp1.toPandas()
            tmp = tmp1.collect()
            tmp = dict(zip([str(i[0]) for i in tmp], [str(i[1]) for i in tmp]))
            df = df.withColumn(col + '_' + main_column + '_ct', df[col].cast('string'))
            map_udf = F.udf(lambda x:tmp.get(x, x))
            df = df.withColumn(col + '_' + main_column + '_ct', map_udf(col + '_' + main_column + '_ct'))
            df = df.withColumn(col + '_' + main_column + '_ct', df[col + '_' + main_column + '_ct'].cast('float'))
            tmp1.unpersist()
            del tmp1,tmp
            gc.collect()
    return df
def main(path = None,run_mode = 'standalone',debug = True):
    if(run_mode == 'standalone'):
        conf = SparkConf()
        # 问题10：默认arrow的优化是关闭的，这里需要手动打开，后面没用topandas了这个设置可以取消，放这里做个记录
        # 问题11：增加了spark.driver.memory，之前一直oom，JGC overhead limit exceeded，java head space之类的，后面把这个显示设置大一点就好了，conf文件里面写的分布式模式才有效，其实standalone模式也需要设置。
        conf.set('spark.sql.execute.arrow.enabled', 'true')
        conf.setAppName('train').setMaster('spark://aigege-OMEN-by-HP-Laptop-15-dc0xxx:7077').set(
            "spark.executor.memory", '3g').set("spark.sql.execution.arrow.enabled", "true").set("spark.sql.crossJoin.enabled", "true")\
            .set("spark.executor.instances", "1").set("spark.executor.cores", "6").set('spark.sql.autoBroadcastJoinThreshold','-1')\
        .set("spark.driver.memory", '8g').set("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1").set('spark.network.timeout','1200')\
        .set('spark.speculation','true').set('spark.sql.shuffle.partitions','500')
        # .set('spark.ui.showConsoleProgress', 'false')


        #问题1：刚开始用Pyspark,发现伪分布式下面，默认读取所有core，比如我的电脑是12，然后executor直接通过spark的配置文件spark-env.sh无法修改，总是只有一个executor，里面的
        # SPARK_WORKER_MEMORY, to set how much total memory workers have to give executors，设置了以后是整个worker的memory，executor默认1024mb，所以下面的.config('spark.executor.memory',
        # '5g')可以设置较大的executor内存,其他的参数类似config('spark.num.executors', '2').config('spark.executor.cores', '6')都可以使用，按需选择即可
        # spark = SparkSession.builder.master('spark://aigege-OMEN-by-HP-Laptop-15-dc0xxx:7077').appName('train').config('spark.executor.memory', '5g').\
        #     config("spark.sql.execution.arrow.enabled", "true").getOrCreate()
        spark = SparkSession.builder. \
            config(conf=conf). \
            getOrCreate()
        # #
        # # # Generate a Pandas DataFrame
        # pdf = pd.DataFrame(np.random.rand(3000000, 3))
        #
        # # Create a Spark DataFrame from a Pandas DataFrame using Arrow
        # df = spark.createDataFrame(pdf)
        #
        # # Convert the Spark DataFrame back to a Pandas DataFrame using Arrow
        # start = time.time()
        # result_pdf = df.toPandas()
        # end = time.time() - start
        # spark.conf.set("spark.sql.crossJoin.enabled", "true")
        # spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    elif(run_mode == 'online'):
        spark = SparkSession.builder.appName(
            'test').getOrCreate()
    with timer("Load data finish:"):
        # LOAD TRAIN
        cols_float = cols.copy()
        cols_float.remove('TransactionID')
        row_nums = 1000
        # 问题2：直接用read.csv的默认参数读取hdfs上面的csv，会导致所有列的类型被读取为string，需要设置inferSchema=True，来自动推测列类型，达到pandas的read_csv的效果
        if debug:
            X_train = spark.read.csv(path=path + 'train_transaction.csv', schema=None, sep=',', header=True,
                                     inferSchema=True).select(cols + ['isFraud'])
            X_train = X_train.sample(fraction=0.1, seed=2020)
        else:
            X_train = spark.read.csv(path=path + 'train_transaction.csv', schema=None, sep=',', header=True,
                                     inferSchema=True).select(cols + ['isFraud'])
        train_id = spark.read.csv(path=path+'train_identity.csv', schema=None, sep=',', header=True,inferSchema=True)
        X_train = X_train.join(train_id, on='TransactionID', how='left')
        # LOAD TEST
        if debug:
            X_test = spark.read.csv(path=path + 'test_transaction.csv', schema=None, sep=',', header=True,
                                    inferSchema=True).select(cols)
            X_test = X_test.sample(fraction=0.1, seed=2020)
        else:
            X_test = spark.read.csv(path=path + 'test_transaction.csv', schema=None, sep=',', header=True,
                                    inferSchema=True).select(cols)
        test_id = spark.read.csv(path=path+'test_identity.csv', schema=None, sep=',', header=True,inferSchema=True)
        X_test = X_test.join(test_id, on='TransactionID', how='left')

        # TARGET
        y_train = X_train.select(*['TransactionID','isFraud'])
        X_train = X_train.drop('isFraud')
        # 修改列类型
        X_test = X_test.select([col(column).cast('string') if dtypes[column]=='string'  else col(column) for column in X_test.columns ])
        X_test = X_test.select([col(column).cast('float') if dtypes[column]=='float'  else col(column) for column in X_test.columns ])
        print((X_test.count(), len(X_test.columns)))

        X_train = X_train.select([col(column).cast('string') if dtypes[column]=='string' else col(column) for column in X_train.columns ])
        X_train = X_train.select([col(column).cast('float') if dtypes[column]=='float'  else col(column) for column in X_train.columns ])
        print((X_train.count(), len(X_train.columns)))


        # X_train = X_train.persist(storageLevel=StorageLevel(True, True, False, False, 2))
        # X_test = X_test.persist(storageLevel=StorageLevel(True, True, False, False, 2))
        # X_train.show()
        # print(X_train.dtypes)
    with timer("Data Preprocess"):
        # NORMALIZE D COLUMNS

        for i in range(1, 16):
            if i in [1, 2, 3, 5, 9]: continue
            # 问题3：Pyspark不能除以numpy.float32,会报错，上网查了下，说是两种type还是不太一样。所以这里直接乘以1.0用python的float，或者改为用np.float64即可
            X_train= X_train.withColumn('D' + str(i),X_train['D' + str(i)] -(X_train['TransactionDT']/ np.float64(24 * 60 * 60)))
            X_test = X_test.withColumn('D' + str(i),X_test['D' + str(i)] - (X_test['TransactionDT']/  np.float64(24 * 60 * 60)))
        col_type = dict(X_train.dtypes)

        # 问题4：Pyspark没找到直接切片的方法，所以用monotonically_increasing_id函数生产index列，作为后续切片的一句，这种方法产生的
        #monotonically_increasing_id不是连续递增，而只是保证每个分区里面的id是单调递增的，所以无法保证很好的切分数据，另一种找到的方法是
        # 创建以下函数
        # def getrows(df, rownums=None):
        #     return df.rdd.zipWithIndex().filter(lambda x: x[1] in rownums).map(lambda x: x[0])
        # 对dataframe切片的函数，比如df[0:100]，跟这种功能类似，rownums可以使用【1,2,3】或者rownums=range(0,X_train.count())来达到
        # 选取指定行或者指定行范围的切片效果。但是这种方法返回的是rdd，返回以后虽然可以使用toDF来达到转换Dataframe的目的，但是有可能因为
        #RDD中元素的内部结构是未知的、不明确的，也就是说每个元素里面有哪些字段，每个字段是什么类型，这些都是不知道的，而DataFrame则要求对
        # 元素的内部结构有完全的知情权。导致出现ValueError: Some of types cannot be determined by the first 100 rows的错误。最后选择
        #的方法是用limit(n)和subtract来达到train和test在和并编码完成以后再分开的效果。
         # LABEL ENCODE AND MEMORY REDUCE

        X_train = X_train.withColumn('train_label',X_train['card1']*0+1)
        X_test = X_test.withColumn('train_label', X_test['card1']*0)

        df_comb = X_train.union(X_test)

        # df_comb.show(100)
        ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain', 'dist1']
        for i,f in enumerate(X_train.columns):
            if f == 'train_label':
                continue
             # FACTORIZE CATEGORICAL VARIABLES
            if (np.str(col_type[f])=='string'):
                print(f)
                # 这里是一开始不熟悉Pyspark api，写的效率比较低，留在这里保存下，和后面写法结果是一致的，只是比较慢
                # df_comb = X_train[[f,'TransactionID']].union(X_test[[f,'TransactionID']])
                # # df_comb = pd.concat([X_train[f],X_test[f]],axis=0)
                # stringIndexer = StringIndexer(inputCol=f, outputCol="encode", handleInvalid="keep",
                # stringOrderType="frequencyDesc")
                # model = stringIndexer.fit(df_comb)
                # label_encode = model.transform(df_comb)
                # label_encode_train = label_encode.limit(X_train.count())
                # label_encode__test = label_encode.exceptAll(label_encode_train)
                # # 问题5：本来生成的编码列直接赋值给原来的列就行了，但是spark不支持这种方式，只能拼接进去以后再删除原始列，然后修改列名了，因为没有concat，
                # #Pyspark里面的concat是直接拼到一列里面，不是pandas那种axis=1然后多出一新列,必须使用join，withColumn只能在本df上面的列上操作后才能直接新增一列，
                # # 不适用于本情况中其他df中的一列新增
                # X_train= X_train.join(label_encode_train[["encode",'TransactionID']],how = 'left',on = 'TransactionID').drop(f).withColumnRenamed('encode',f)
                # X_test = X_test.join(label_encode__test[["encode",'TransactionID']],how = 'left',on = 'TransactionID').drop(f).withColumnRenamed('encode',f)
                stringIndexer = StringIndexer(inputCol=f, outputCol=f+"_encode", handleInvalid="keep",
                stringOrderType="frequencyDesc")
                model = stringIndexer.fit(df_comb)
                df_comb = model.transform(df_comb)
                # SHIFT ALL NUMERICS POSITIVE. SET NAN to -1
            elif f not in ['TransactionAmt', 'TransactionDT','TransactionID']:
                print(f)
                min_tmp = df_comb.agg({f: "min"}).collect()[0][0]
                df_comb = df_comb.withColumn(f,(df_comb[f]-min_tmp).cast('float'))
                df_comb = df_comb.fillna({f:-1})

        rename_col = [name + '_encode' for name in str_type]
        rename_col = dict(zip(rename_col, str_type))
        # 问题6：drop不能像pandas那样，直接drop['a','b'],前面的加*，或者直接一个一个写出，例如：‘a'，’b'
        df_comb = df_comb.drop(*str_type)
        # 问题7：对列名批量重命名，一开始是准备使用withcolumnrename，然后将变量名生成的字典传进去作为参数，类似{‘old nmae’：‘new name’，。。。}
        # 但是发现这个api这样用无效，后面就在stackoverflow找了下面这种曲线救国的方式来批量重命名，效果和pandas rename一样。
        df_comb = df_comb.select([col(c).alias(rename_col.get(c, c)) for c in df_comb.columns])
        # X_train = df_comb.limit(X_train.count())
        X_train = df_comb.filter('train_label == 1')
        # X_test = df_comb.exceptAll(X_train)
        X_test = df_comb.filter('train_label == 0')
        print(X_train.count(), len(X_train.columns))
        print(X_test.count(), len(X_test.columns))

        df_comb.unpersist()
        del df_comb
        gc.collect()
    with timer("Feature engineering"):
        # TRANSACTION AMT CENTS
        X_train = X_train.withColumn('cents',X_train['TransactionAmt'] - F.floor(X_train['TransactionAmt']))
        X_test = X_test.withColumn('cents',X_test['TransactionAmt'] - F.floor(X_test['TransactionAmt']))
        df_comb = X_train.union(X_test)
        # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
        df_comb = encode_FE(df_comb, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
        # COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN
        df_comb = encode_CB(df_comb,'card1', 'addr1')
        df_comb = encode_CB(df_comb,'card1_addr1', 'P_emaildomain')
        # FREQUENCY ENOCDE
        df_comb = encode_FE(df_comb, ['card1_addr1', 'card1_addr1_P_emaildomain'])
        # GROUP AGGREGATE
        df_comb = encode_AG(['TransactionAmt', 'D9', 'D11'], ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'],df_comb,
                  ['mean', 'std'], usena=True)

        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        # 这里是模仿下面两列注释的pandas的写法，因为没找到pandas那样自带的属性，只能自己写自定义udf函数调用。
        ## X_train['DT_M'] = X_train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        date_udf = F.udf(lambda x: START_DATE + datetime.timedelta(seconds=x), TimestampType())
        df_comb = df_comb.withColumn('DT_M',date_udf('TransactionDT'))

        ## X_train['DT_M'] = (X_train['DT_M'].dt.year - 2017) * 12 + X_train['DT_M'].dt.month
        month_udf = F.udf(lambda x: x.month, IntegerType())
        year_udf = F.udf(lambda x:x.year, IntegerType())
        df_comb = df_comb.withColumn('DT_M', (year_udf('DT_M')-2017)*12+month_udf('DT_M'))

        # AGGREGATE
        df_comb = df_comb.withColumn('day',df_comb['TransactionDT']/ (24 * 60 * 60))
        df_comb = df_comb.withColumn('uid', F.format_string('%s_%s', df_comb['card1_addr1'].cast('string'), F.floor(df_comb['day']-df_comb['D1'])))
        # df_comb = encode_AG2(['P_emaildomain','dist1','DT_M','id_02','cents'], ['uid'], df_comb)

        # FREQUENCY ENCODE UID
        df_comb = encode_FE(df_comb,['uid'])
        # AGGREGATE
        df_comb = encode_AG(['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['uid'], df_comb, ['mean', 'std'], fillna=True, usena=True)
        # AGGREGATE
        df_comb = encode_AG(['C' + str(x) for x in range(1, 15) if x != 3], ['uid'], df_comb, ['mean'], fillna=True,
                  usena=True)
        # AGGREGATE
        df_comb = encode_AG(['M' + str(x) for x in range(1, 10)], ['uid'], df_comb, ['mean'], fillna=True, usena=True)
        # AGGREGATE
        df_comb = encode_AG2(['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents'], ['uid'], df_comb)
        # AGGREGATE
        df_comb = encode_AG(['C14'], ['uid'], df_comb, ['std'], fillna=True, usena=True)
        # AGGREGATE
        df_comb = encode_AG2(['C13', 'V314'], ['uid'], df_comb)
        # AGGREATE
        df_comb = encode_AG2(['V127', 'V136', 'V309', 'V307', 'V320'], ['uid'], df_comb)
        # NEW FEATURE
        df_comb = df_comb.withColumn('outsider15', F.when(F.abs(df_comb['D1'] - df_comb['D15'])>3,1).otherwise(0))
        df_comb.show(100)
        print('outsider15')
        cols1 = list(df_comb.columns)
        cols1.remove('TransactionDT')
        cols1.remove('train_label')
        for c in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
            cols1.remove(c)
        for c in ['DT_M', 'day', 'uid']:
            cols1.remove(c)

        # FAILED TIME CONSISTENCY TEST
        for c in ['C3', 'M5', 'id_08', 'id_33']:
            cols1.remove(c)
        for c in ['card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']:
            cols1.remove(c)
        for c in ['id_' + str(x) for x in range(22, 28)]:
            cols1.remove(c)
        print('NOW USING THE FOLLOWING', len(cols1), 'FEATURES.')
        # X_train = df_comb.limit(X_train.count())
        # X_test = df_comb.exceptAll(X_train)
        X_train = df_comb.filter('train_label == 1')
        X_test = df_comb.filter('train_label == 0')
        print(X_test.count())
        print(X_test.dtypes)
        print(X_train.dtypes)
        print((X_test.count(), len(X_test.columns)))
        print((X_train.count(), len(X_train.columns)))
        print(X_train.columns)
        # X_train.write.mode('overwrite').csv(path+'X_train.csv',header='true')
        # X_test.write.mode('overwrite').csv(path + 'X_test.csv',header='true')
        XY_train = X_train.join(y_train, on='TransactionID', how='left')
        XY_train.show(10)
        # XY_train.write.mode('overwrite').csv(path + 'XY_train.csv', header='true')
        del X_train,X_test,
        gc.collect()

    with timer("model train test:"):
        # LOAD TRAIN
        cols_float = cols.copy()
        cols_float.remove('TransactionID')
        row_nums = 1000
        # 问题2：直接用read.csv的默认参数读取hdfs上面的csv，会导致所有列的类型被读取为string，需要设置inferSchema=True，来自动推测列类型，达到pandas的read_csv的效果
        # if debug:
        #     XY_train = spark.read.csv(path=path + 'XY_train.csv', schema=None, sep=',', header=True,
        #                              inferSchema=True)
        #     XY_train = XY_train.sample(fraction=0.1, seed=2020)
        # else:
        #     XY_train = spark.read.csv(path=path + 'XY_train.csv', schema=None, sep=',', header=True,
        #                              inferSchema=True)
        # X_test = spark.read.csv(path=path+'X_test.csv', schema=None, sep=',', header=True,inferSchema=True)
        print(XY_train.columns)
        print(XY_train.count())
        cols1 = list(XY_train.columns)
        cols1.remove('TransactionDT')
        for c in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
            cols1.remove(c)
        for c in ['DT_M', 'day', 'uid']:
            print(c)
            cols1.remove(c)

        # FAILED TIME CONSISTENCY TEST
        for c in ['C3', 'M5', 'id_08', 'id_33']:
            cols1.remove(c)
        for c in ['card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']:
            cols1.remove(c)
        for c in ['id_' + str(x) for x in range(22, 28)]:
            cols1.remove(c)

        for c in ['train_label', 'ProductCD', 'TransactionID']:
            print(c)
            cols1.remove(c)

        print('NOW USING THE FOLLOWING', len(cols1), 'FEATURES.')
        XY_train = XY_train.fillna(0)
        train, test = XY_train.randomSplit([0.8, 0.2], seed=2019)
        print('~~~~~~~~~~~~~',test.count())
        # train.show(100)
        for i in cols1:
            if i not in train.columns:
                print(i)
        cols1.remove('isFraud')
        featurizer = VectorAssembler(
            inputCols=cols1,
            outputCol='features'
        )
        # 问题12：此处后面的‘features’列对应于VectorAssembler里面的inputCols里面所有的列，全部会合并到‘features’里面变为一个向量，spark训练要求使用此种格式传入，不然会报错，找不到features列，
        # 剩余的列用于训练后评估指标。验证集的处理方式一样
        train_data = featurizer.transform(train)['TransactionID', 'features','isFraud']
        featurizer = VectorAssembler(
            inputCols=cols1,
            outputCol='features'
        )
        test_data = featurizer.transform(test)['TransactionID','features','isFraud']
        print(test_data.count(),len(test_data.columns))
        from mmlspark.lightgbm import LightGBMClassifier
        model = LightGBMClassifier(learningRate=0.05,
                                   numIterations=400,isUnbalance = True
                                   numLeaves=31,labelCol='isFraud').fit(train_data)
        print('train finish')
        predict = model.transform(test_data)['prediction','isFraud','probability','rawPrediction']
        print('ppredict finish')
        predict = predict.toPandas()
        def dense_to_num(x):
            a = x.values[1]
            return a
        predict['probability'] = predict['probability'].apply(dense_to_num)
        print(roc_auc_score(predict['isFraud'],predict['probability']))


if __name__ == "__main__":
    path = "hdfs://localhost:9000/user/kaggle_fraud_detection/data/"
    with timer("Full feature select run"):
        main(path = path,run_mode = 'standalone',debug = False)