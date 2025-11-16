from dvclive.live import Live
from xgboost import XGBClassifier
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score


def main():
    loaded_model = XGBClassifier()
    loaded_model.load_model(os.path.join("models", "model.pkl"))

    X_test = pd.read_csv(os.path.join(
        "transformed_data", "X_test.csv"), index_col=0)
    y_test = pd.read_csv(os.path.join(
        "transformed_data", "y_test.csv"), index_col=0)

    prediction = loaded_model.predict(X_test)

    with Live("eval") as live:
        live.log_metric("accuracy", accuracy_score(y_test, prediction))
        live.log_metric("precision", precision_score(
            y_test, prediction, average="micro"))
        live.log_metric("f1", f1_score(y_test, prediction, average="micro"))


if __name__ == "__main__":
    main()
