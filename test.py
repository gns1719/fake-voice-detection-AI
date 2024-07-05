# test.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from torch.utils.data import DataLoader
from preprocess import Config
from train import CustomDataset
from model import RNN
import random
import os

CONFIG = Config()

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

        if train_mode and 'label' in df.columns:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode and labels:
        return np.array(features), np.array(labels)
    return np.array(features)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_test_data(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    test_mel = get_mel_spectrogram_feature(test_df, train_mode=False)
    test_dataset = CustomDataset(test_mel, None)
    return test_dataset, test_df['id']

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for features in test_loader:
            features = features.float().to(device)
            output = model(features)
            predictions.append(output.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    return predictions

if __name__ == "__main__":
    seed_everything(CONFIG.SEED)
    
    test_dataset, test_ids = load_test_data('./test.csv')
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda')
    
    model = RNN().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    predictions = predict(model, test_loader, device)
    
    submission = pd.DataFrame(predictions, columns=['fake', 'real'])
    submission['id'] = test_ids
    submission = submission[['id', 'fake', 'real']]  # Rearrange columns to ['id', 'fake', 'real']
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv.")