import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    uri = "models:/cancer-classifier-prod@staging"
    model = mlflow.pyfunc.load_model(uri)

    data = load_breast_cancer(as_frame=True)
    print(f"Loading {uri}")

    for label, name in enumerate(data.target_names):
        index = data.target[data.target == label].index[0]
        sample = data.data.loc[[index]]

        predicted = int(model.predict(sample)[0])
        predicted_name = data.target_names[predicted]

        print(
            f"row={index} actual={name} predicted={predicted_name} "
            f"correct={predicted == label}"
        )


if __name__ == "__main__":
    load_and_predict()