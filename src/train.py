import os
from dvc.api import params_show
import pandas as pd
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D
)
import tensorflow.keras as keras
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


def build_model(input_dim=64, seq_len=50, model_name="GRU"):
    """
    Build a simple LSTM model
    """

    options = {
        "GRU": GRU(64, return_sequences=False, input_shape=(seq_len, input_dim)),
        "LSTM": LSTM(64, return_sequences=False, input_shape=(seq_len, input_dim)),

        "BiGRU": Bidirectional(GRU(64, return_sequences=False,
                               input_shape=(seq_len, input_dim))),
        "BiLSTM": Bidirectional(LSTM(64, return_sequences=False,
                                     input_shape=(seq_len, input_dim))),

        "Conv1D": Sequential([
            Conv1D(64, 5, activation="relu", input_shape=(seq_len, input_dim)),
            MaxPooling1D(2),
            Conv1D(128, 5, activation="relu"),
            GlobalAveragePooling1D(),
        ]),

        "TCN": Sequential([
            Conv1D(64, 3, dilation_rate=1, padding="causal",
                   activation="relu", input_shape=(seq_len, input_dim)),
            Conv1D(64, 3, dilation_rate=2, padding="causal", activation="relu"),
            GlobalAveragePooling1D(),
        ]),

        "MLP_Mixer": Sequential([
            Dense(128, activation="gelu", input_shape=(seq_len, input_dim)),
            Dense(128, activation="gelu"),
            GlobalAveragePooling1D(),
        ])
    }

    model = Sequential([
        options[model_name],
        Dense(32, activation="relu"),
        Dense(10, activation="softmax")     # adjust to your task
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def main():
    os.makedirs(OUT_FILE, exist_ok=True)

    params = params_show()
    seq_length = params["transform"]["seq_length"]
    n_mels = params["transform"]["n_mels"]
    model_name = params["train"]["model_name"]
    batch_size = params["train"]["batch_size"]
    epochs = params["train"]["epochs"]

    dataset = np.load("sequences/train.npz")
    print(dataset)
    X = dataset["arr_0"]
    y = dataset["arr_1"]

    print("Final dataset shape:", X.shape)

    model = build_model(input_dim=n_mels, seq_len=seq_length,
                        model_name=model_name)
    model.summary()

    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    model.save(f"{OUT_FILE}{model_name}.keras")


if __name__ == "__main__":
    main()
