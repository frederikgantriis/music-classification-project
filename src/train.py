import os
from dvc.api import params_show
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dropout
)

IN_FILE = "transformed_data/"
OUT_FILE = "models/"


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
            Dropout(0.3),
            Conv1D(64, 3, dilation_rate=2, padding="causal", activation="relu"),
            GlobalAveragePooling1D(),
        ]),

        "MLP_Mixer": Sequential([
            Dense(128, activation="gelu", input_shape=(seq_len, input_dim)),
            Dropout(0.3),
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
