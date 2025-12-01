import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from dvclive.live import Live
from dvc.api import params_show


def get_waveform(filename, y, sr):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    return plt.gcf()


def get_spectrogram(filename, y, sr):
    D = np.abs(librosa.stft(y))
    DB = librosa.amplitude_to_db(D, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(DB, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram (STFT) {filename}")
    plt.tight_layout()

    return plt.gcf()


def get_mel_spectrogram(filename, y, sr, n_mels):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel-Spectrogram {filename}")
    plt.tight_layout()

    return plt.gcf()


def main():
    base_url = os.path.join("raw-data", "genres_original")

    params = params_show()["transform"]
    n_mels = params["n_mels"]

    # Load varies examples
    filenames = [
        os.path.join(base_url, "classical", "classical.00000.wav"),
        os.path.join(base_url, "blues", "blues.00000.wav"),
        os.path.join(base_url, "rock", "rock.00000.wav"),
        os.path.join(base_url, "pop", "pop.00000.wav"),
        os.path.join(base_url, "jazz", "jazz.00000.wav"),
        os.path.join(base_url, "country", "country.00000.wav"),
        os.path.join(base_url, "disco", "disco.00000.wav"),
        os.path.join(base_url, "hiphop", "hiphop.00000.wav"),
        os.path.join(base_url, "metal", "metal.00000.wav"),
        os.path.join(base_url, "reggae", "reggae.00000.wav"),
    ]

    with Live("visualise") as live:
        for filename in filenames:
            y, sr = librosa.load(filename, sr=None)

            # Log waveform
            live.log_image(f"{filename}-waveform.png",
                           get_waveform(filename, y, sr))

            # Log spectrogram
            live.log_image(f"{filename}-spectrogram.png",
                           get_spectrogram(filename, y, sr))

            # Log mel-spectrogram
            live.log_image(f"{filename}-mel_spectrogram.png",
                           get_mel_spectrogram(filename, y, sr, n_mels))


if __name__ == "__main__":
    main()
