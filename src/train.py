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
from codecarbon import EmissionsTracker
from dvclive.live import Live

IN_FILE = "transformed_data/"
OUT_FILE = "models/"


def build_model(input_dim=64, seq_len=50, model_name="GRU"):
    """
    Build a simple LSTM model
    """

    options = {
        "GRU": GRU(128, return_sequences=False, input_shape=(seq_len, input_dim)),
        "LSTM": LSTM(128, return_sequences=False, input_shape=(seq_len, input_dim)),

        "BiGRU": Bidirectional(GRU(128, return_sequences=False,
                               input_shape=(seq_len, input_dim))),
        "BiLSTM": Bidirectional(LSTM(128, return_sequences=False,
                                     input_shape=(seq_len, input_dim))),

        "Conv1D": Sequential([
            Conv1D(128, 5, activation="relu",
                   input_shape=(seq_len, input_dim)),
            MaxPooling1D(2),
            Conv1D(128, 5, activation="relu"),
            GlobalAveragePooling1D(),
        ]),

        "TCN": Sequential([
            Conv1D(128, 3, dilation_rate=1, padding="causal",
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

    tracker = EmissionsTracker()

    tracker.start()

    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    emissions = tracker.stop()

    model.save(f"{OUT_FILE}{model_name}.keras")

    with Live("emissions") as live:
        live.log_param("CO2", emissions)


if __name__ == "__main__":
    main()
