#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2019.12.20
# os.environ['PYSPARK_PYTHON'] = '/home/aigege/.conda/envs/py3-conda/bin/python3.6'
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions  as f
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import Row
import gc
import time
from contextlib import contextmanager
import warnings
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
    dtypes[c] = 'float32'
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def main(path = None,run_mode = 'standalone'):
    if(run_mode == 'standalone'):
        spark = SparkSession.builder.master('spark://aigege-OMEN-by-HP-Laptop-15-dc0xxx:7077').appName('train').getOrCreate()
    elif(run_mode == 'online'):
        spark = SparkSession.builder.appName(
            'test').getOrCreate()
    with timer("Load data finish:"):
        # LOAD TRAIN
        X_train = spark.read.csv(path=path+'train_transaction.csv', schema=None, sep=',', header=True).select(cols + ['isFraud'])
        train_id = spark.read.csv(path=path+'train_identity.csv', schema=None, sep=',', header=True)
        train_id.show()
        X_train = X_train.join(train_id, on='TransactionID', how='left')
        print((X_train.count(), len(X_train.columns)))
        # LOAD TEST
        X_test = spark.read.csv(path=path+'test_transaction.csv', schema=None, sep=',', header=True).select(cols)
        test_id = spark.read.csv(path=path+'test_identity.csv', schema=None, sep=',', header=True)
        X_test = X_test.join(test_id, on='TransactionID', how='left')
        print((X_test.count(), len(X_test.columns)))




if __name__ == "__main__":
    path = "hdfs://localhost:9000/user/kaggle_fraud_detection/data/"
    with timer("Full feature select run"):
        main(path = path,run_mode = 'standalone')