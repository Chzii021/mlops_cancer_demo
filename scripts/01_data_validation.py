import argparse

import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data(min_balance=0.20):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")

        df = load_breast_cancer(as_frame=True).frame
        rows, cols = df.shape
        classes = df["target"].nunique()
        missing = int(df.isna().sum().sum())
        balance = float(df["target"].value_counts(normalize=True).min())

        passed = (
            rows == 569
            and cols == 31
            and classes == 2
            and missing == 0
            and balance >= min_balance
        )

        status = "Success" if passed else "Failed"

        mlflow.log_params({
            "num_classes": classes,
            "min_class_balance": min_balance,
            "validation_status": status,
        })

        mlflow.log_metrics({
            "num_rows": rows,
            "num_cols": cols,
            "missing_values": missing,
            "class_balance": balance,
        })

        print(f"Dataset shape: {rows} rows, {cols} columns")
        print(f"Number of classes: {classes}")
        print(f"Missing values: {missing}")
        print(f"Class balance: {balance:.4f}")
        print(f"Validation status: {status}")

        if not passed:
            raise SystemExit("Data validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-balance", type=float, default=0.20)
    args = parser.parse_args()
    validate_data(args.min_balance)