import os
import pandas as pd
from dvc.api import params_show
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import shutil

INDEX_FILE = "index_files/"
OUT_FILE = "transformed_data/"


def main():
    os.makedirs(INDEX_FILE, exist_ok=True)
    os.makedirs(OUT_FILE, exist_ok=True)
    os.makedirs(OUT_FILE + "train/", exist_ok=True)
    os.makedirs(OUT_FILE + "test/", exist_ok=True)

    params = params_show()["transform"]

    df = pd.read_csv(os.path.join(
        "raw-data",
        "features_30_sec.csv"
    ))

    # Selecting labels
    y = df[['label']]

    # Converting cateogries to numeric
    y['label_int'] = df['label'].astype('category').cat.codes
    y = y.drop(["label"], axis=1)
    y = y.rename(columns={"label_int": "label"})

    # Selecting filename
    X = df["filename"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"]
    )

    print(X, y)

    for filename in X:
        label = df.loc[df["filename"] == filename, "label"].iloc[0]

        shutil.copy(
            os.path.join(
                "raw-data",
                "genres_original",
                label,
                filename
            ),
            os.path.join(
                OUT_FILE,
                "train/",
                filename
            )
        )

    pd.DataFrame(X_train).to_csv(os.path.join(INDEX_FILE, "X_train.csv"))
    pd.DataFrame(X_test).to_csv(os.path.join(INDEX_FILE, "X_test.csv"))
    pd.DataFrame(y_train).to_csv(os.path.join(INDEX_FILE, "y_train.csv"))
    pd.DataFrame(y_test).to_csv(os.path.join(INDEX_FILE, "y_test.csv"))


if __name__ == "__main__":
    main()
