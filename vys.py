#!/usr/bin/env python3

import pyspark
import pyspark.sql as psql
import pyspark.sql.functions as psqlfunc
from pyspark import SparkContext
import pandas as pd
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql import SQLContext
from operator import add
from operator import itemgetter, attrgetter
import os
import glob
import re

#pattern = re.compile("(moment_filt|angle|emg2|imu_real|velocity)\.csv$")
#print(bool( pattern.search("moment_filt.csv")))

# create the Spark Session
spark = psql.SparkSession.builder.getOrCreate()
# create the Spark Context
sc = spark.sparkContext

for s in glob.glob("AB*"):
    for a in glob.glob(s + "/*"):
        activity_path = glob.glob(a + "/*activity_flag.csv")
        #print(activity_file)
        assert(len(activity_path) == 1)
        #these = ["moment_filt", "angle", "emg2", "imu_real", "velocity"]
        pattern = re.compile("(moment_filt|angle|emg2|imu_real|velocity)\.csv$")
        #pattern = re.compile(r'.csv$')
        activity_df = spark.read.option("header", True).csv(activity_path)
        for d in filter(lambda name: pattern.search(name), list(set(glob.glob(a + "/*.csv")) - set(activity_path))):
            print(d)
            df = spark.read.option("header", True).csv(d)
            #df.show()
            activity_df = activity_df.join(df, ["time"])
            #activity_df.show()
            #exit(1)
        df.write.csv(a + "/aggregate")


#lines = sc.textFile()
