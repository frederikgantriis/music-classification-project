from dvclive.live import Live
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from dvc.api import params_show
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def translate_numbers(list, params):
    translated_list = []

    for item in list:
        translated_list.append(params[item])

    return translated_list


def main():
    params = params_show()

    model_name = params["train"]["model_name"]

    dict = params_show("dict/params.yaml")

    model = load_model(f"models/{model_name}.keras")

    dataset = np.load("sequences/test.npz")

    X = dataset["arr_0"]
    y = dataset["arr_1"].flatten()

    prediction_prob = model.predict(X)
    prediction = np.argmax(prediction_prob, axis=1)

    prediction_translated = translate_numbers(prediction, dict)
    y_translated = translate_numbers(y, dict)

    cm = confusion_matrix(y_translated, prediction_translated)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    with Live("eval") as live:
        live.log_metric("accuracy", accuracy_score(y, prediction))
        live.log_metric("precision", precision_score(
            y, prediction, average="micro"))
        live.log_image("confusion_matrix.png", plt.gcf())


if __name__ == "__main__":
    main()
