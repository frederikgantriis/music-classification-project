import os
from dvc.api import params_show
import pandas as pd
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
import audioread
from tqdm import trange

IN_FILE = "transformed_data/"
OUT_FILE = "models/"


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


def build_model(input_dim=64, seq_len=50):
    """
    Build a simple LSTM model
    """
    model = Sequential([
        GRU(64, return_sequences=False, input_shape=(seq_len, input_dim)),
        Dense(32, activation="relu"),
        Dense(10, activation="softmax")     # adjust to your task
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def main():
    os.makedirs(OUT_FILE, exist_ok=True)

    params = params_show()["train"]
    seq_length = params["sequence_length"]
    n_mels = params["n_mels"]
    stride = params["stride"]

    X_files = pd.read_csv(os.path.join("index_files", "X_train.csv"))
    y_labels = pd.read_csv(os.path.join("index_files", "y_train.csv"))

    print(X_files)
    print(y_labels)

    labels = y_labels["label"].values

    folder = os.path.join("transformed_data", "train")
    features = load_audio_files(folder, n_mels)
    X, y = create_sequences(features, labels, seq_len=seq_length, step=stride)

    print("FINAL Y", y)

    print("Final dataset shape:", X.shape)
    # Example: (1200 sequences, 50 timesteps, 64 features)

    # Dummy target labels for demonstration:

    model = build_model(input_dim=64, seq_len=seq_length)
    model.summary()

    model.fit(X, y, batch_size=32, epochs=5)


if __name__ == "__main__":
    main()
