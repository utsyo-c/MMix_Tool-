import os
os.environ['SPARK_HOME'] = r'C:\Users\AnnaSim\Spark'
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk1.8.0_202'
os.environ['HADOOP_HOME'] = r'C:\Users\AnnaSim\hadoop'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'jupyter'
os.environ['PYSPARK_DRIVER_PYTHON_OPTS'] = 'lab'
os.environ['PYSPARK_PYTHON'] = 'python'

import pyspark
# Import PySpark
from pyspark.sql import SparkSession


# spark = SparkSession.builder \
#     .appName("LocalTest") \
#     .master("local[*]") \
#     .getOrCreate()

# df = spark.range(5)
# df.show()