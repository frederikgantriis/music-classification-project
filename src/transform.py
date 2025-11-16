import os
import pandas as pd
from dvc.api import params_show
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

OUT_FILE = "transformed_data/"


def main():
    os.makedirs(OUT_FILE, exist_ok=True)

    params = params_show()["transform"]

    df = pd.read_csv(os.path.join(
        "raw-data",
        "features_30_sec.csv"
    ))

    y = df[['label']]
    y['label_int'] = df['label'].astype('category').cat.codes

    y = y.drop(["label"], axis=1)

    y = y.rename(columns={"label_int": "label"})

    X = df.drop(["label", "filename"], axis=1)

    print(X, y)

    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)

    # new data frame with the new scaled data.
    X = pd.DataFrame(np_scaled, columns=cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"]
    )

    print(X_train, X_test, y_train, y_test)

    pd.DataFrame(X_train).to_csv(os.path.join(OUT_FILE, "X_train.csv"))
    pd.DataFrame(X_test).to_csv(os.path.join(OUT_FILE, "X_test.csv"))
    pd.DataFrame(y_train).to_csv(os.path.join(OUT_FILE, "y_train.csv"))
    pd.DataFrame(y_test).to_csv(os.path.join(OUT_FILE, "y_test.csv"))


if __name__ == "__main__":
    main()
