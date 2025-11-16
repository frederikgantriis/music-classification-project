import os
from dvc.api import params_show
from xgboost import XGBClassifier
import pandas as pd
import pickle as pkl

IN_FILE = "transformed_data/"
OUT_FILE = "models/"


def main():
    os.makedirs(OUT_FILE, exist_ok=True)

    params = params_show()["train"]

    X_train = pd.read_csv(os.path.join(IN_FILE, "X_train.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(IN_FILE, "y_train.csv"), index_col=0)

    model = XGBClassifier()

    model.fit(X_train, y_train)

    model.save_model("models/model.pkl")


if __name__ == "__main__":
    main()
