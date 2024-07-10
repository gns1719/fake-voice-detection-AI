# preprocess.py

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

class Config:
    SR = 32000
    N_MELS = 128
    FMAX = 8000
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 5
    LR = 3e-4
    SEED = 42
    MAX_LEN = 1000

def load_audio(file_path, sr=Config.SR, duration=None):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    if Config.MAX_LEN:
        if len(audio) > Config.MAX_LEN:
            audio = audio[:Config.MAX_LEN]
        else:
            audio = np.pad(audio, (0, Config.MAX_LEN - len(audio)))
    return audio

def extract_melspectrogram(audio, sr=Config.SR, n_mels=Config.N_MELS, fmax=Config.FMAX):
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    return melspec_db

def preprocess_data(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio = load_audio(row['path'])
        melspec = extract_melspectrogram(audio)
        np.save(os.path.join(output_dir, f"{row['id']}.npy"), melspec)

    return df

if __name__ == "__main__":
    train_df = preprocess_data("train.csv", "processed_train")
    test_df = preprocess_data("test.csv", "processed_test")