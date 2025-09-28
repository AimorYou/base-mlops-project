import os
import logging
import warnings

import pandas as pd

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


MY_NAME = "Arseny"
MY_SURNAME = "Pivovarov"
EXPERIMENT_NAME = f"{MY_SURNAME}_{MY_NAME[0]}"
MLFLOW_RUN_ID = "aimoryou"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


def prepare_data():
    titanic = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    df = titanic.frame

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

    logging.info(f"Data prepared: X_train={X_train_trans.shape}, X_test={X_test_trans.shape}")

    return X_train_trans, X_test_trans, y_train, y_test


def train_and_log(name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name, nested=True) as child:
        child_run_id = child.info.run_id
        logging.info(f"Started child run (model_name={name}, id={child_run_id})")

        mlflow.log_params({f"param_{k}": str(v) for k, v in model.get_params().items()})
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)

        sig = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
            signature=sig,
            input_example=X_train.head(3),
        )

        logging.info(f"Child run finished (model_name={name}, id={child_run_id}): metrics={metrics}")

        return {"run_id": child_run_id, "metrics": metrics, "model": model, "model_name": name}


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = prepare_data()

    named_models = [
        ("LogisticRegression", LogisticRegression(max_iter=500, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    with mlflow.start_run(run_name=MLFLOW_RUN_ID) as parent:
        parent_run_id = parent.info.run_id
        logging.info(f"Started parent run: {MLFLOW_RUN_ID} (id={parent_run_id})")

        run_results = []
        for name, model in named_models:
            run_result = train_and_log(name, model, X_train, y_train, X_test, y_test)
            run_results.append(run_result)

        best = None
        best_f1 = -1
        for res in run_results:
            f1 = res["metrics"]["f1"]

            if f1 > best_f1:
                best_f1 = f1
                best = res

        best_run_id = best["run_id"]
        best_model_name = f"{best['model_name']}_{MY_SURNAME}"
        logging.info(f"Best run chosen: model_name={best_model_name}, run_id={best_run_id}: f1={best_f1}")

        model_uri = f"runs:/{best_run_id}/model"
        reg_model = mlflow.register_model(model_uri=model_uri, name=best_model_name)
        best_model_version = reg_model.version
        logging.info(f"Registered model: name={best_model_name}, version={best_model_version}")

        client = MlflowClient()
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_model_version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logging.info(f"Transitioned model {best_model_name} version {best_model_version} to Staging")


if __name__ == "__main__":
    main()
