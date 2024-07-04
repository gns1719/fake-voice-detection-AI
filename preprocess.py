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

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

def get_mel_spectrogram_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CONFIG.N_MELS, fmax=CONFIG.FMAX)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Pad or truncate to MAX_LEN
        if mel_spectrogram_db.shape[1] > CONFIG.MAX_LEN:
            mel_spectrogram_db = mel_spectrogram_db[:, :CONFIG.MAX_LEN]
        else:
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, CONFIG.MAX_LEN - mel_spectrogram_db.shape[1])), mode='constant')
        
        features.append(mel_spectrogram_db)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return np.array(features), np.array(labels)
    return np.array(features)

train_mel, train_labels = get_mel_spectrogram_feature(train, True)
val_mel, val_labels = get_mel_spectrogram_feature(val, True)

# Save preprocessed data
print("Saving preprocessed data...")
np.save('train_mel.npy', np.array(train_mel))
np.save('val_mel.npy', np.array(val_mel))
np.save('train_labels.npy', np.array(train_labels))
np.save('val_labels.npy', np.array(val_labels))
print("Preprocessed data saved successfully.")