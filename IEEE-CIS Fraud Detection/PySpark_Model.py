#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.12.20
# os.environ['PYSPARK_PYTHON'] = '/home/aigege/.conda/envs/py3-conda/bin/python3.6'
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
import numpy as np
import gc
import time
from contextlib import contextmanager
import warnings
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.stat import Summarizer

import os
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
for c in str_type: dtypes[c] = 'string'
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# FREQUENCY ENCODE TOGETHER
def encode_FE(df, cols):
    for col in cols:
        # 生成对应的每个值的出现频率字典
        tmp = df.groupBy(col).count().orderBy('count',ascending = False)
        tmp_sum = tmp.select(F.sum('count')).collect()[0][0]
        tmp = tmp.withColumn('count',tmp['count']/tmp_sum)
        tmp = tmp[col,'count'].collect()
        tmp = dict(zip([i[0] for i in tmp],[i[1] for i in tmp]))
        tmp[-1] = -1
        nm = col + '_FE'
        df = df.withColumn(nm, df[col])
        df = df.replace(to_replace=tmp, subset=[nm])
        print(nm, ', ', end='')
    return df
# LABEL ENCODE
# def encode_LE(col, df, verbose=True):
#     df_comb = pd.concat([train[col], test[col]], axis=0)
#     df_comb, _ = df_comb.factorize(sort=True)
#     nm = col
#     if df_comb.max() > 32000:
#         train[nm] = df_comb[:len(train)].astype('int32')
#         test[nm] = df_comb[len(train):].astype('int32')
#     else:
#         train[nm] = df_comb[:len(train)].astype('int16')
#         test[nm] = df_comb[len(train):].astype('int16')
#     del df_comb;
#     x = gc.collect()
#     if verbose: print(nm, ', ', end='')
# COMBINE FEATURES
def encode_CB(df,col1, col2, ):
    nm = col1 + '_' + col2
    df= df.withColumn(nm,df[col1].cast('string') + '_' + df[col2].cast('string'))
    df.show(100)
    # encode_LE(nm, verbose=False)
    print(nm, ', ', end='')


def main(path = None,run_mode = 'standalone'):
    if(run_mode == 'standalone'):
        #问题1：刚开始用Pyspark,发现伪分布式下面，默认读取所有core，比如我的电脑是12，然后executor直接通过spark的配置文件spark-env.sh无法修改，总是只有一个executor，里面的
        # SPARK_WORKER_MEMORY, to set how much total memory workers have to give executors，设置了以后是整个worker的memory，executor默认1024mb，所以下面的.config('spark.executor.memory',
        # '5g')可以设置较大的executor内存,其他的参数类似config('spark.num.executors', '2').config('spark.executor.cores', '6')都可以使用，按需选择即可
        spark = SparkSession.builder.master('spark://aigege-OMEN-by-HP-Laptop-15-dc0xxx:7077').appName('train').config('spark.executor.memory', '5g').\
            getOrCreate()
        spark.conf.set("spark.sql.crossJoin.enabled", "true")
    elif(run_mode == 'online'):
        spark = SparkSession.builder.appName(
            'test').getOrCreate()
    with timer("Load data finish:"):
        # LOAD TRAIN
        cols_float = cols.copy()
        cols_float.remove('TransactionID')
        # 问题2：直接用read.csv的默认参数读取hdfs上面的csv，会导致所有列的类型被读取为string，需要设置inferSchema=True，来自动推测列类型，达到pandas的read_csv的效果
        X_train = spark.read.csv(path=path+'train_transaction.csv', schema=None, sep=',', header=True,inferSchema=True).select(cols + ['isFraud'])
        train_id = spark.read.csv(path=path+'train_identity.csv', schema=None, sep=',', header=True,inferSchema=True)
        print(X_train.dtypes)
        X_train = X_train.join(train_id, on='TransactionID', how='left')
        # LOAD TEST
        X_test = spark.read.csv(path=path+'test_transaction.csv', schema=None, sep=',', header=True,inferSchema=True).select(cols)
        test_id = spark.read.csv(path=path+'test_identity.csv', schema=None, sep=',', header=True,inferSchema=True)
        X_test = X_test.join(test_id, on='TransactionID', how='left')
        # TARGET
        y_train = X_train[['isFraud']]
        X_train = X_train.drop('isFraud')
        # 修改列类型
        X_test = X_test.select([col(column).cast('string') if dtypes[column]=='string'  else col(column) for column in X_test.columns ])
        X_test = X_test.select([col(column).cast('float') if dtypes[column]=='float'  else col(column) for column in X_test.columns ])
        print((X_test.count(), len(X_test.columns)))
        X_train = X_train.select([col(column).cast('string') if dtypes[column]=='string' else col(column) for column in X_train.columns ])
        X_train = X_train.select([col(column).cast('float') if dtypes[column]=='float'  else col(column) for column in X_train.columns ])
        print((X_train.count(), len(X_train.columns)))
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
        print(X_train.count(), len(X_train.columns))
        print(X_test.count(), len(X_test.columns))
        df_comb = X_train.union(X_test)
        for i,f in enumerate(['addr1', 'card1', 'card2', 'card3', 'P_emaildomain']):
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
            elif(np.str(col_type[f])!='string') and  f not in ['TransactionAmt', 'TransactionDT','TransactionID']:
                print(f)
                min_tmp = df_comb.agg({f: "min"}).collect()[0][0]
                df_comb = df_comb.withColumn(f,df_comb[f]-min_tmp)
                df_comb = df_comb.fillna({f:-1})
        rename_col = [name + '_encode' for name in str_type]
        rename_col = dict(zip(rename_col, str_type))
        # 问题6：drop不能像pandas那样，直接drop['a','b'],前面的加*，或者直接一个一个写出，例如：‘a'，’b'
        df_comb = df_comb.drop(*str_type)
        # 问题7：对列名批量重命名，一开始是准备使用withcolumnrename，然后也讲变量名生成的字典传进去作为参数，类似{‘old nmae’：‘new name’，。。。}
        # 但是发现这个api这样用无效，后面就在stackoverflow找了下面这种曲线救国的方式来批量重命名，效果和pandas rename一样。
        df_comb = df_comb.select([col(c).alias(rename_col.get(c, c)) for c in df_comb.columns])
        X_train = df_comb.limit(X_train.count())
        X_test = df_comb.exceptAll(X_train)
    with timer("Feature engineering"):
        # TRANSACTION AMT CENTS
        X_train = X_train.withColumn('cents',X_train['TransactionAmt'] - F.floor(X_train['TransactionAmt']))
        X_test = X_test.withColumn('cents',X_test['TransactionAmt'] - F.floor(X_test['TransactionAmt']))
        df_comb = X_train.union(X_test)
        # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
        # df_comb = encode_FE(df_comb, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
        # COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN
        encode_CB(df_comb,'card1', 'addr1')

        a = 1


if __name__ == "__main__":
    path = "hdfs://localhost:9000/user/kaggle_fraud_detection/data/"
    with timer("Full feature select run"):
        main(path = path,run_mode = 'standalone')