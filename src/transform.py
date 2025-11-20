import os
import pandas as pd
from dvc.api import data, params_show
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import shutil
import numpy as np
import librosa
import audioread
from tqdm import trange
from dvclive import Live

INDEX_FILE = "index_files/"
OUT_FILE = "transformed_data/"


def load_audio_files(folder, n_mels):
    """
    Load audio files, load spectrogram and return
    """
    features = []

    pbar = trange(len(os.listdir(folder)))

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            pbar.set_description(f"Working on {path}")

            try:
                # Load audio
                audio, sr = librosa.load(path, sr=16000, mono=True)
                # Extract mel spectrogram (time x features)
                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=n_mels,
                    hop_length=256
                )

                # Convert to log scale
                mel = librosa.power_to_db(mel).T  # shape: (timesteps, 64)

                features.append(mel)
            except audioread.exceptions.NoBackendError:
                print("Skipping", path, ": Corrupt File Error")

            pbar.update()

    pbar.close()

    return features


def create_sequences(features_list, labels, step=1, seq_len=50):
    """
    Split each file's feature matrix into sequences
    """
    sequences = []
    seq_labels = []

    pbar = trange(len(features_list))

    for feat, label in zip(features_list, labels):
        for i in range(0, len(feat) - seq_len, step):
            seq = feat[i:i + seq_len]
            sequences.append(seq)
            seq_labels.append(label)

        pbar.update()

    # (num_sequences, seq_len, 64)

    pbar.close()
    return np.array(sequences), np.array(seq_labels)


def process_dataset(X, y, df, n_mels, seq_length, stride, dataset_name):
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
                dataset_name,
                filename
            )
        )

    labels = y["label"].values

    folder = os.path.join("transformed_data", dataset_name)
    features = load_audio_files(folder, n_mels)
    X, y = create_sequences(
        features, labels, seq_len=seq_length, step=stride)

    np.savez(os.path.join("sequences", f"{dataset_name}.npz"), X, y)


def main():
    os.makedirs(INDEX_FILE, exist_ok=True)
    os.makedirs(OUT_FILE + "train/", exist_ok=True)
    os.makedirs(OUT_FILE + "test/", exist_ok=True)
    os.makedirs("sequences/", exist_ok=True)

    params = params_show("params.yaml")["transform"]

    n_mels = params["n_mels"]
    seq_length = params["seq_length"]
    stride = params["stride"]

    df = pd.read_csv(os.path.join(
        "raw-data",
        "features_30_sec.csv"
    ))

    # Selecting labels
    y = df[['label']]

    # Converting cateogries to numeric
    y['label_int'] = df['label'].astype('category').cat.codes

    with Live("dict") as live:
        live.log_params(
            dict(enumerate(y['label'].astype('category').cat.categories)))

    y = y.drop(["label"], axis=1)
    y = y.rename(columns={"label_int": "label"})

    # Selecting filename
    X = df["filename"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"]
    )

    print(X, y)

    process_dataset(X_train, y_train, df, n_mels, seq_length, stride, "train")
    process_dataset(X_test, y_test, df, n_mels, seq_length, stride, "test")


if __name__ == "__main__":
    main()
