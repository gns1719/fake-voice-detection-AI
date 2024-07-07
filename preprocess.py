# preprocess.py

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split

class Config:
    SR = 32000
    N_MELS = 128
    FMAX = 8000
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 10
    LR = 1e-4
    SEED = 42
    MAX_LEN = 1000

CONFIG = Config()

def get_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=CONFIG.SR)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CONFIG.N_MELS, fmax=CONFIG.FMAX)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    if mel_spectrogram_db.shape[1] > CONFIG.MAX_LEN:
        mel_spectrogram_db = mel_spectrogram_db[:, :CONFIG.MAX_LEN]
    else:
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, CONFIG.MAX_LEN - mel_spectrogram_db.shape[1])), mode='constant')
    
    return mel_spectrogram_db

def preprocess_and_save(df, output_path):
    with h5py.File(output_path, 'w') as h5:
        mel_dataset = h5.create_dataset('mel', shape=(len(df), CONFIG.N_MELS, CONFIG.MAX_LEN), dtype=np.float32)
        label_dataset = h5.create_dataset('label', shape=(len(df), CONFIG.N_CLASSES), dtype=np.float32)
        
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            mel = get_mel_spectrogram(row['path'])
            mel_dataset[i] = mel
            
            label = np.zeros(CONFIG.N_CLASSES, dtype=np.float32)
            label[0 if row['label'] == 'fake' else 1] = 1
            label_dataset[i] = label

if __name__ == "__main__":
    df = pd.read_csv('./train.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)
    
    print("Preprocessing training data...")
    preprocess_and_save(train_df, 'train_data.h5')
    
    print("Preprocessing validation data...")
    preprocess_and_save(val_df, 'val_data.h5')
    
    print("Preprocessing completed.")