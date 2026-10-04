import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "cancer-classifier-prod"
MIN_ACCURACY = 0.95
MIN_ROC_AUC = 0.98


def train_evaluate_register(preprocessing_run_id, C=10.0):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Breast Cancer - Model Training")

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")

        mlflow.log_params({
            "preprocessing_run_id": preprocessing_run_id,
            "C": C,
            "random_state": 42,
            "max_iter": 10000,
        })

        folder = Path(download_artifacts(
            run_id=preprocessing_run_id,
            artifact_path="processed_data",
        ))

        train = pd.read_csv(folder / "train.csv")
        test = pd.read_csv(folder / "test.csv")

        X_train = train.drop(columns="target")
        y_train = train["target"]
        X_test = test.drop(columns="target")
        y_test = test["target"]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=C, random_state=42, max_iter=10000,
            )),
        ])

        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)
        positive_index = list(pipeline.classes_).index(1)
        proba = pipeline.predict_proba(X_test)[:, positive_index]

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)

        print(f"Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")
        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="cancer_classifier_pipeline",
            input_example=X_train.head(5),
        )

        passed = acc >= MIN_ACCURACY and auc >= MIN_ROC_AUC
        mlflow.set_tag("quality_gate", "Passed" if passed else "Failed")

        if passed:
            registered = mlflow.register_model(
                model_info.model_uri, MODEL_NAME,
            )

            MlflowClient().set_registered_model_alias(
                name=MODEL_NAME,
                alias="staging",
                version=registered.version,
            )

            print(f"Registered {MODEL_NAME} version {registered.version}")
            print(f"Set alias @staging -> version {registered.version}")
        else:
            print("Gate failed: model logged but NOT registered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessing_run_id")
    parser.add_argument("C", nargs="?", type=float, default=10.0)
    args = parser.parse_args()

    train_evaluate_register(args.preprocessing_run_id, args.C)