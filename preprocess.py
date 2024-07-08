# preprocess.py

import librosa
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import torch
import os
from tqdm import tqdm
import warnings
import h5py

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    SR = 32000
    N_MELS = 128
    FMAX = 8000
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 32  # Reduced batch size
    N_EPOCHS = 5
    LR = 3e-4
    SEED = 42
    MAX_LEN = 1000
    CHUNK_SIZE = 1000  # Number of samples to process at once

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)

df = pd.read_csv('./train.csv')
train, val = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)

def get_mel_spectrogram_feature(row):
    y, sr = librosa.load(row['path'], sr=CONFIG.SR)
    
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CONFIG.N_MELS, fmax=CONFIG.FMAX)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    if mel_spectrogram_db.shape[1] > CONFIG.MAX_LEN:
        mel_spectrogram_db = mel_spectrogram_db[:, :CONFIG.MAX_LEN]
    else:
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, CONFIG.MAX_LEN - mel_spectrogram_db.shape[1])), mode='constant')
    
    return mel_spectrogram_db

def process_data(df, h5_file, dataset_name, mode='w'):
    with h5py.File(h5_file, mode) as f:
        feature_dataset = f.create_dataset(f'{dataset_name}_mel', shape=(len(df), CONFIG.N_MELS, CONFIG.MAX_LEN),
                                           dtype='float32', chunks=True, maxshape=(None, CONFIG.N_MELS, CONFIG.MAX_LEN))
        label_dataset = f.create_dataset(f'{dataset_name}_labels', shape=(len(df), CONFIG.N_CLASSES),
                                         dtype='float32', chunks=True, maxshape=(None, CONFIG.N_CLASSES))
        
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            mel_spectrogram_db = get_mel_spectrogram_feature(row)
            feature_dataset[i] = mel_spectrogram_db
            
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            label_dataset[i] = label_vector

if __name__ == "__main__":
    process_data(train, 'train_data.h5', 'train', mode='w')
    process_data(val, 'train_data.h5', 'val', mode='a')
    print("Preprocessed data saved successfully.")