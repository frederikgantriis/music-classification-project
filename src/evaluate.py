from dvclive.live import Live
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
from tensorflow.keras.models import load_model
from dvc.api import params_show
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Load parameters
    params = params_show()

    # Load modelname
    model_name = params["train"]["model_name"]

    # Load model
    model = load_model(f"models/{model_name}.keras")

    # Load test dataset
    dataset = np.load("sequences/test.npz")

    # Load x and y
    X = dataset["arr_0"]
    y = dataset["arr_1"].flatten()

    # Get predictions
    prediction_prob = model.predict(X)

    # Convert predictions to the prediction with the highest probability
    prediction = np.argmax(prediction_prob, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y, prediction)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    with Live("eval") as live:
        # Log metrics
        live.log_metric("accuracy", accuracy_score(y, prediction))
        live.log_metric("precision", precision_score(
            y, prediction, average="macro"))
        live.log_metric("F1 Score", f1_score(y, prediction, average="macro"))
        live.log_metric("recall", recall_score(y, prediction, average="macro"))

        # Log confusion matrix
        live.log_image("confusion_matrix.png", plt.gcf())


if __name__ == "__main__":
    main()
