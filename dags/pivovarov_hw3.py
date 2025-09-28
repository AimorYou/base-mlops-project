import io
import os
import logging
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
MY_NAME = ""
MY_SURNAME = ""
MLFLOW_EXPERIMENT_NAME = f"{MY_SURNAME}{MY_NAME[0]}_Final"

S3_BUCKET = Variable.get("S3_BUCKET")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")


# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_csv(buf)


def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)


# -----------------
# Таски
# -----------------
def init_pipeline(**context):
    ts = datetime.utcnow().isoformat()
    logging.info(f"pipeline_start={ts}")


def collect_data(**context):
    ### Ваш код здесь.


def split_and_preprocess(**context):
    ### Ваш код здесь.


def train_and_log_mlflow(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    AWS_ENDPOINT_URL,
    **context,
):
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_ENDPOINT_URL"] = AWS_ENDPOINT_URL
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

    ### Ваш код здесь.


def serve_model(**context):
    ### Ваш код здесь.


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw33",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    init_pipeline = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    collect_data = PythonOperator(task_id="collect_data", python_callable=collect_data)
    split_and_preprocess = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    train_and_log_mlflow = PythonOperator(
        task_id="train_and_log_mlflow",
        python_callable=train_and_log_mlflow,
        op_kwargs={
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": "ru-central1",
            "AWS_ENDPOINT_URL": "https://storage.yandexcloud.net",
        },
    )
    serve_model = PythonOperator(task_id="serve_model", python_callable=serve_model) # Можете заменить на любой другой оператор!

    init_pipeline >> collect_data >> split_and_preprocess >> train_and_log_mlflow >> serve_model
