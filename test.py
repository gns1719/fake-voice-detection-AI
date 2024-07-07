# test.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from torch.utils.data import Dataset, DataLoader
from model import CRNN, Config
import random
import os

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_mel_spectrogram_feature(df):
    features = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CONFIG.N_MELS, fmax=CONFIG.FMAX)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if mel_spectrogram_db.shape[1] > CONFIG.MAX_LEN:
            mel_spectrogram_db = mel_spectrogram_db[:, :CONFIG.MAX_LEN]
        else:
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, CONFIG.MAX_LEN - mel_spectrogram_db.shape[1])), mode='constant')
        
        features.append(mel_spectrogram_db)

    return np.array(features)

class TestDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]).unsqueeze(0)

def load_test_data(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    test_mel = get_mel_spectrogram_feature(test_df)
    return TestDataset(test_mel), test_df['id']

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for features in tqdm(test_loader, desc="Predicting"):
            features = features.to(device)
            output = model(features)
            predictions.append(output.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    return predictions

if __name__ == "__main__":
    seed_everything(CONFIG.SEED)
    
    test_dataset, test_ids = load_test_data('./test.csv')
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda')
    
    model = CRNN().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    predictions = predict(model, test_loader, device)
    
    submission = pd.DataFrame(predictions, columns=['fake', 'real'])
    submission['id'] = test_ids
    submission = submission[['id', 'fake', 'real']]
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv.")