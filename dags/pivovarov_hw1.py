import io
import json
import logging
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


AWS_CONN_ID = "S3_CONNECTION"
S3_BUCKET = Variable.get("S3_BUCKET")
MY_NAME = "Arseny"
MY_SURNAME = "Pivovarov"

S3_KEY_RAW_DATA = f"{MY_SURNAME}/raw_data.csv"
S3_KEY_X_TRAIN_DATA = f"{MY_SURNAME}/X_train.csv"
S3_KEY_X_TEST_DATA = f"{MY_SURNAME}/X_test.csv"
S3_KEY_Y_TRAIN_DATA = f"{MY_SURNAME}/y_train.csv"
S3_KEY_Y_TEST_DATA = f"{MY_SURNAME}/y_test.csv"
S3_KEY_MODEL_OBJECT = f"{MY_SURNAME}/model.pickle"
S3_KEY_MODEL_METRICS = f"{MY_SURNAME}/model_metrics.json"
S3_KEY_PIPELINE_METRICS = f"{MY_SURNAME}/pipeline_metrics.json"


def write_to_s3(hook: S3Hook, obj, bucket: str, key: str, file_format: str = "pkl") -> None:
    buf = io.BytesIO()
    if file_format == "csv":
        obj.to_csv(buf, index=False)
    elif file_format == "pkl":
        joblib.dump(obj, buf)
    elif file_format == "json":
        buf.write(json.dumps(obj).encode("utf-8"))
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)


def read_from_s3(hook: S3Hook, bucket: str, key: str, file_format: str = "pkl"):
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)

    if file_format == "csv":
        return pd.read_csv(buf)
    elif file_format == "pkl":
        return joblib.load(buf)
    elif file_format == "json":
        return json.load(buf)
    else:
        raise ValueError(f"Unsupported format: {file_format}")


# ---- TASKS -----


def init_pipeline(**context):
    start_ts = datetime.utcnow().isoformat()
    logging.info(f"Запуск пайплайна: {start_ts}")
    context["ti"].xcom_push(key="pipeline_start", value=start_ts)


def collect_data(**context):
    titanic = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    df = titanic.frame

    hook = S3Hook(AWS_CONN_ID)
    write_to_s3(hook, df, S3_BUCKET, S3_KEY_RAW_DATA, file_format="csv")
    logging.info("Raw data saved to S3")


def split_and_preprocess(**context):
    hook = S3Hook(AWS_CONN_ID)
    df = read_from_s3(hook, S3_BUCKET, S3_KEY_RAW_DATA, file_format="csv")
    
    df["age"].fillna(df["age"].median(), inplace=True)
    df["fare"].fillna(df["fare"].median(), inplace=True)
    df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
    
    cat_feats = ["sex", "embarked", "pclass"]
    num_feats = ["age", "fare", "sibsp", "parch"]

    X = df[cat_feats + num_feats]
    y = df["survived"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(), cat_feats)
    ])
    
    preprocessor.fit(X_train)
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_trans = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_trans = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)
        
    write_to_s3(hook, X_train_trans, S3_BUCKET, S3_KEY_X_TRAIN_DATA, file_format="csv")
    write_to_s3(hook, X_test_trans, S3_BUCKET, S3_KEY_X_TEST_DATA, file_format="csv")
    write_to_s3(hook, y_train, S3_BUCKET, S3_KEY_Y_TRAIN_DATA, file_format="csv")
    write_to_s3(hook, y_test, S3_BUCKET, S3_KEY_Y_TEST_DATA, file_format="csv")
    
    logging.info("Train/test split and preprocessing done")


def train_model(**context):
    hook = S3Hook(AWS_CONN_ID)

    X_train = read_from_s3(hook, S3_BUCKET, S3_KEY_X_TRAIN_DATA, file_format="csv")
    y_train = read_from_s3(hook, S3_BUCKET, S3_KEY_Y_TRAIN_DATA, file_format="csv")

    start_train = datetime.now()
    logging.info(f"Training started at {start_train.isoformat()}")
    
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    end_train = datetime.now()
    logging.info(f"Training finished at {end_train.isoformat()}")
    
    write_to_s3(hook, model, S3_BUCKET, S3_KEY_MODEL_OBJECT, file_format="pkl")

    context["ti"].xcom_push(key="train_start", value=start_train.isoformat())
    context["ti"].xcom_push(key="train_end", value=end_train.isoformat())

    logging.info("Model trained and saved to S3")


def collect_metrics_model(**context):
    hook = S3Hook(AWS_CONN_ID)

    X_test = read_from_s3(hook, S3_BUCKET, S3_KEY_X_TEST_DATA, file_format="csv")
    y_test = read_from_s3(hook, S3_BUCKET, S3_KEY_Y_TEST_DATA, file_format="csv")

    model = read_from_s3(hook, S3_BUCKET, S3_KEY_MODEL_OBJECT, file_format="pkl")

    y_pred = model.predict(X_test)

    model_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    write_to_s3(hook, model_metrics, S3_BUCKET, S3_KEY_MODEL_METRICS, file_format="json")

    logging.info(f"Model evaluation done: {json.dumps(model_metrics, indent=4)}")


def collect_metrics_pipeline(**context):
    pipeline_end_ts = datetime.utcnow().isoformat()
    pipeline_start_ts = context["ti"].xcom_pull(key="pipeline_start")
    
    train_start_ts = context["ti"].xcom_pull(key="train_start")
    train_end_ts = context["ti"].xcom_pull(key="train_end")

    pipeline_metrics = {
        "train_start_time": train_start_ts,
        "train_end_time": train_end_ts,
        "train_duration_seconds": (
            datetime.fromisoformat(train_end_ts) - datetime.fromisoformat(train_start_ts)
        ).total_seconds(),
        "pipeline_start_time": pipeline_start_ts,
        "pipeline_end_time": pipeline_end_ts,
        "pipeline_duration_seconds": (
            datetime.fromisoformat(pipeline_end_ts) - datetime.fromisoformat(pipeline_start_ts)
        ).total_seconds(),
    }

    hook = S3Hook(AWS_CONN_ID)
    
    write_to_s3(hook, pipeline_metrics, S3_BUCKET, S3_KEY_PIPELINE_METRICS, file_format="json")

    logging.info(f"Pipeline finished: {json.dumps(pipeline_metrics, indent=4)}")


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw1",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    t1 = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    t2 = PythonOperator(task_id="collect_data", python_callable=collect_data)
    t3 = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    t4 = PythonOperator(task_id="train_model", python_callable=train_model)
    t5 = PythonOperator(task_id="collect_metrics_model", python_callable=collect_metrics_model)
    t6 = PythonOperator(task_id="collect_metrics_pipeline", python_callable=collect_metrics_pipeline)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
