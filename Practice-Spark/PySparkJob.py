import os
import shutil

import click
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_COLS = ['ad_id', 'target_audience_count',
             'has_video', 'ad_cost', 'ad_cost_type']
RESULT_PROPORTIONS = [.75, .25]
RESULTS_DIRNAMES = ['train', 'test']


def get_spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


def get_base_frame(data_frame):
    base_frame = data_frame[BASE_COLS].drop_duplicates()
    base_frame = base_frame.selectExpr(
        'ad_id',
        'target_audience_count',
        'has_video',
        "cast(ad_cost_type = 'CPM' as int) as is_cpm",
        "cast(ad_cost_type = 'CPC' as int) as is_cpc",
        'ad_cost')
    return base_frame


def get_additional_info(data_frame):
    additional_info = data_frame.groupby('ad_id').agg(
        F.countDistinct('date').alias('day_count'),
        F.round((
            F.sum((data_frame.event == 'click').astype('int')) /
            F.sum((data_frame.event == 'view').astype('int'))
        ), 6).alias('CTR')
    )
    return additional_info


def process_data(data_frame):
    base_frame = get_base_frame(data_frame)
    additional_info = get_additional_info(data_frame)
    full_frame = base_frame.join(additional_info, on='ad_id', how='leftouter')
    results = full_frame.randomSplit(RESULT_PROPORTIONS)
    return results


def save_results(results, target_path, overwrite=False):
    os.makedirs(target_path, exist_ok=True)
    for dirname, result in zip(RESULTS_DIRNAMES, results):
        target_dir = os.path.join(target_path, dirname)
        if os.path.isdir(target_dir):
            if any(os.scandir(target_dir)) and not overwrite:
                raise(IOError('Target directories isn\'t empty!'))
            shutil.rmtree(target_dir)
        result.write.parquet(target_dir)


@click.command()
@click.option('-o', '--overwrite', is_flag=True, default=False, help='Overwrite data in results path (if exists)')
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('target_path')
def main(overwrite, input_path, target_path):
    """
    Process data\n
    input_path     Path to initial data frame\n
    target_path    Path to write results
    """
    spark = get_spark_session()
    data_frame = spark.read.parquet(input_path)
    results = process_data(data_frame)
    save_results(results, target_path, overwrite)


if __name__ == "__main__":
    main()
