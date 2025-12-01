import os
import pandas as pd
from dvc.api import params_show
from sklearn.model_selection import train_test_split
from dvclive.live import Live
from utils import process_dataset

OUT_FILE = "transformed_data/"


def main():
    # Make directories for output files
    os.makedirs("sequences/", exist_ok=True)
    os.makedirs(OUT_FILE + "train/", exist_ok=True)
    os.makedirs(OUT_FILE + "test/", exist_ok=True)

    os.remove(os.path.join("raw-data", "genres_original",
              "jazz", "jazz.00054.wav"))

    # Load parameters
    params = params_show("params.yaml")["transform"]

    # Define each parameter
    n_mels = params["n_mels"]
    seq_length = params["seq_length"]
    stride = params["stride"]

    # Load dataset overview
    df = pd.read_csv(os.path.join(
        "raw-data",
        "features_30_sec.csv"
    ))

    df = df[df["filename"] != "jazz.00054.wav"]

    # Selecting labels
    y = df[['label']]

    # Converting cateogries to numeric
    y['label_int'] = df['label'].astype('category').cat.codes

    with Live("dict") as live:
        # Log all categories
        live.log_params(
            dict(enumerate(y['label'].astype('category').cat.categories)))

    # Define label
    y = y.drop(["label"], axis=1)
    y = y.rename(columns={"label_int": "label"})

    # Selecting filename
    X = df["filename"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"]
    )

    # Process each dataset
    process_dataset(X_train, y_train, df, n_mels, seq_length, stride, "train")
    process_dataset(X_test, y_test, df, n_mels, seq_length, stride, "test")


if __name__ == "__main__":
    main()
