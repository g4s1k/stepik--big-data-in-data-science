from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions, DataFrame


class ColsFilter(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def transform(df: DataFrame) -> DataFrame:
        result = df.select('ad_id', 'features', functions.col('ctr').alias('label'))
        return result