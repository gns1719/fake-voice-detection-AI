import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from torch.utils.data import DataLoader
from preprocess import Config, seed_everything, get_mel_spectrogram_feature
from train import CustomDataset
from model import CNNWithAttention
import h5py

CONFIG = Config()

def load_test_data(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    
    with h5py.File('test_data.h5', 'w') as f:
        feature_dataset = f.create_dataset('test_mel', shape=(len(test_df), CONFIG.INPUT_CHANNELS, CONFIG.N_MELS, CONFIG.MAX_LEN),
                                           dtype='uint8', chunks=True)
        
        for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df))):
            rgb_spectrogram = get_mel_spectrogram_feature(row)
            feature_dataset[i] = rgb_spectrogram
    
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
    
    model = CNNWithAttention().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    predictions = predict(model, test_loader, device)
    
    submission = pd.DataFrame(predictions, columns=['fake', 'real'])
    submission['id'] = test_ids
    submission = submission[['id', 'fake', 'real']]  # Rearrange columns to ['id', 'fake', 'real']
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv.")