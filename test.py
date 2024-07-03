# test.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from model import MLP, seed_everything
from preprocess import Config

def get_mel_spectrogram_feature(df):
    features = []
    for _, row in tqdm(df.iterrows(), desc="Extracting Mel-Spectrogram features"):
        file_path = f"./test/{row['id']}.ogg"
        y, sr = librosa.load(file_path, sr=32000)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MELS, fmax=Config.FMAX)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_db = np.mean(mel_spectrogram_db.T, axis=0)
        features.append(mel_spectrogram_db)
    return features

def load_model(model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, features, device):
    model.to(device)
    predictions = []
    with torch.no_grad():
        for feature in tqdm(features, desc="Making predictions"):
            feature = torch.FloatTensor(feature).unsqueeze(0).to(device)
            output = model(feature)
            predictions.append(output.cpu().numpy().squeeze())
    return predictions

def main():
    seed_everything(Config.SEED)
    device = torch.device('cuda')

    # Load the sample submission file
    sample_submission = pd.read_csv('sample_submission.csv')

    # Load and preprocess test data
    test_features = get_mel_spectrogram_feature(sample_submission)

    # Load the trained model
    model = load_model('best_model.pth')

    # Make predictions
    predictions = predict(model, test_features, device)

    # Update the sample submission with predictions
    sample_submission.iloc[:, 1:] = predictions

    # Save the results
    sample_submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()
