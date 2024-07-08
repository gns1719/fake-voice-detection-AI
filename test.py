# test.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from torch.utils.data import DataLoader
from preprocess import Config, seed_everything
from train import CustomDataset
from model import RNNWithAttention
import h5py

CONFIG = Config()

def get_mel_spectrogram_feature(row):
    y, sr = librosa.load(row['path'], sr=CONFIG.SR)
    
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CONFIG.N_MELS, fmax=CONFIG.FMAX)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    if mel_spectrogram_db.shape[1] > CONFIG.MAX_LEN:
        mel_spectrogram_db = mel_spectrogram_db[:, :CONFIG.MAX_LEN]
    else:
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, CONFIG.MAX_LEN - mel_spectrogram_db.shape[1])), mode='constant')
    
    return mel_spectrogram_db

def load_test_data(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    
    with h5py.File('test_data.h5', 'w') as f:
        feature_dataset = f.create_dataset('test_mel', shape=(len(test_df), CONFIG.N_MELS, CONFIG.MAX_LEN),
                                           dtype='float32', chunks=True)
        
        for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df))):
            mel_spectrogram_db = get_mel_spectrogram_feature(row)
            feature_dataset[i] = mel_spectrogram_db
    
    test_dataset = CustomDataset('test_data.h5', 'test', test_mode=True)
    return test_dataset, test_df['id'].tolist()

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
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RNNWithAttention().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    predictions = predict(model, test_loader, device)
    
    submission = pd.DataFrame(predictions, columns=['fake', 'real'])
    submission['id'] = test_ids
    submission = submission[['id', 'fake', 'real']]  # Rearrange columns to ['id', 'fake', 'real']
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv.")