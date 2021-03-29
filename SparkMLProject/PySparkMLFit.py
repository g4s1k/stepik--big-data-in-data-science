import sys
import os
from typing import List
import json

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession, DataFrame

from colsfilter import ColsFilter


# Используйте как путь куда сохранить модель
MODEL_PATH = 'spark_ml_model'

PARAMS = {
    'maxDepth': [_ for _ in range(5, 10)],
    'maxBins': [_ for _ in range(110, 131, 2)]
}


def load_data(spark: SparkSession, train_data: str, test_data:str) -> List[DataFrame]:
    train_df = spark.read.parquet(train_data)
    test_df = spark.read.parquet(test_data)
    return [train_df, test_df]


def init_pipeline(inputColumns: str) -> Pipeline:
    model = DecisionTreeRegressor(seed=42)
    vectorizer = VectorAssembler(inputCols=inputColumns, outputCol='features')
    cols_filter = ColsFilter()
    pipeline = Pipeline(stages=[vectorizer, cols_filter, model])
    return pipeline


def get_best_model(pipeline: Pipeline, evaluator: RegressionEvaluator, train_df: DataFrame, test_df: DataFrame) -> dict:
    model = pipeline.getStages()[-1]
    model_params = ParamGridBuilder()\
                    .addGrid(model.maxDepth, PARAMS['maxDepth'])\
                    .addGrid(model.maxBins, PARAMS['maxBins'])\
                    .build()
    best_model_finder = TrainValidationSplit(estimator=pipeline,
                                            estimatorParamMaps=model_params,
                                            evaluator=evaluator,
                                            trainRatio=0.7,
                                            seed=42)
    best_model = best_model_finder.fit(train_df).bestModel
    result = dict()
    result['best model'] = best_model
    for param in PARAMS:
        result[param] = best_model.stages[-1].getOrDefault(param)
    best_model_res = best_model.transform(test_df)
    rmse = evaluator.evaluate(best_model_res)
    result['RMSE'] = round(rmse,4)
    return result


def save_result(result: dict):
    result.pop('best model').save(MODEL_PATH)
    params_path = os.path.join(MODEL_PATH, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(result, f)


def process(spark: SparkSession, train_data: str, test_data:str):
    train_df, test_df = load_data(spark, train_data, test_data)
    features_cols = list(train_df.columns[1:-1])
    features_cols.remove('is_cpm')
    pipeline = init_pipeline(inputColumns=features_cols)
    evaluator = RegressionEvaluator()
    training_result = get_best_model(pipeline, evaluator, train_df, test_df)
    save_result(training_result)
    print(training_result)


def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
