# preprocess.py

import librosa
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import os

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda')

class Config:
    SR = 32000
    N_MELS = 128
    FMAX = 8000
    # Dataset
    ROOT_FOLDER = './'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 5
    LR = 3e-4
    # Others
    SEED = 42

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED) # Seed 고정

df = pd.read_csv('./train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

def get_mel_spectrogram_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CONFIG.N_MELS, fmax=CONFIG.FMAX)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_db = np.mean(mel_spectrogram_db.T, axis=0)
        features.append(mel_spectrogram_db)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

train_mel, train_labels = get_mel_spectrogram_feature(train, True)
val_mel, val_labels = get_mel_spectrogram_feature(val, True)

class CustomDataset(Dataset):
    def __init__(self, mel, label):
        self.mel = mel
        self.label = label

    def __len__(self):
        return len(self.mel)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.mel[idx], self.label[idx]
        return self.mel[idx]
    
train_dataset = CustomDataset(train_mel, train_labels)
val_dataset = CustomDataset(val_mel, val_labels)

# Save preprocessed data using pandas
print("Saving preprocessed data...")

# Convert to DataFrames
train_mel_df = pd.DataFrame(train_mel)
train_labels_df = pd.DataFrame(train_labels)
val_mel_df = pd.DataFrame(val_mel)
val_labels_df = pd.DataFrame(val_labels)

# Save to CSV
train_mel_df.to_csv('train_mel.csv', index=False)
train_labels_df.to_csv('train_labels.csv', index=False)
val_mel_df.to_csv('val_mel.csv', index=False)
val_labels_df.to_csv('val_labels.csv', index=False)

print("Preprocessed data saved successfully.")
