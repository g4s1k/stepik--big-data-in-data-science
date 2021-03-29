import sys

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

#necessary class to pipeline object load
from colsfilter import ColsFilter

# Используйте как путь откуда загрузить модель
MODEL_PATH = 'spark_ml_model'


def process(spark: SparkSession, input_file: str, output_file: str):
    model = PipelineModel.load(MODEL_PATH)
    df = spark.read.parquet(input_file)
    result = model.transform(df).select('ad_id', 'prediction')
    result.write.csv(output_file)



def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
