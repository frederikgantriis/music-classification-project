from tqdm import trange
import os
import os
import shutil
import numpy as np
import librosa
import audioread
from tqdm import trange


def load_audio_files(folder, filenames, n_mels):
    """
    Load audio files, load spectrogram and return
    """
    features = []

    # Create progressbar
    pbar = trange(len(filenames))

    # For each file, load melspectrogram
    for file in filenames:
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

            # Scaling the input
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)

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
    """
    Load each file, and move it into either train or test folder
    """

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
                "transformed_data",
                dataset_name,
                filename
            )
        )

    labels = y["label"].values

    folder = os.path.join("transformed_data", dataset_name)

    # Load melspectrogram
    features = load_audio_files(folder, X, n_mels)

    # Create sequences
    X, y = create_sequences(
        features, labels, seq_len=seq_length, step=stride)

    # Save as numpy objects
    np.savez(os.path.join("sequences", f"{dataset_name}.npz"), X, y)
