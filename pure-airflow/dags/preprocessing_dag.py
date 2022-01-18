from airflow import DAG
from airflow.models import BaseOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

from src.nodes import *

with DAG(
    "preprocessing-dag",
    start_date=datetime(2019, 1, 1),
    max_active_runs=3,
    schedule_interval=timedelta(hours=1),  # https://airflow.apache.org/docs/stable/scheduler.html#dag-runs
    catchup=False # enable if you don't want historical dag runs to run
) as dag:

    clean_data_operator = PythonOperator(
        task_id = "clean_data_task",
        python_callable=clean_data
    )
    
    make_transformer_operator = PythonOperator(
        task_id = "make_transformer_task",
        python_callable=make_transformer
    )
    
    split_data_operator = PythonOperator(
        task_id = "split_data_task",
        python_callable=split_data
    )
    
    # train_linear_regression_model_operator = PythonOperator(
    #     task_id = "train_linear_regression_model_task",
    #     python_callable=train_linear_regression_model
    # )
    
    # predict_operator = PythonOperator(
    #     task_id = "predict_task",
    #     python_callable=predict
    # )
    
    # evaluate_model_operator = PythonOperator(
    #     task_id = "evaluate_model_task",
    #     python_callable=evaluate_model
    # )

    clean_data_operator >> make_transformer_operator

    make_transformer_operator >> split_data_operator