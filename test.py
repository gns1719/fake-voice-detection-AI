# test.py
import argparse
import os

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
from model import CNN
from preprocess import Config
import preprocess


def get_mel_spectrogram_feature(df, batch_size=32):
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Extracting Mel-Spectrogram features"):
        batch_features = []
        for _, row in df.iloc[start_idx:start_idx + batch_size].iterrows():
            file_path = os.path.join("./test", f"{row['id']}.ogg")
            try:
                y, sr = librosa.load(file_path, sr=Config.SR)
                mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=Config.N_MELS, fmax=Config.FMAX)
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                mel_spectrogram_db = np.mean(mel_spectrogram_db.T, axis=0)
                batch_features.append(mel_spectrogram_db)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                batch_features.append(np.zeros(Config.N_MELS))  # Handle error by appending a zero feature
        yield batch_features


def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()
    return model


def predict(model, features, device):
    model.to(device)
    predictions = []
    with torch.no_grad():
        for feature in tqdm(features, desc="Making predictions"):
            # 단일 입력처리가 아니라 배치입력처리로 전환하여 unsqueeze를 본 코드에서 제거함
            feature = torch.FloatTensor(feature).to(device)
            outputs = model(feature)
            predictions.extend(outputs.cpu().numpy())
    return predictions


def main(args):
    preprocess.seed_everything(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # Load the sample submission file
    sample_submission = pd.read_csv('sample_submission.csv')

    # Load and preprocess test data
    feature_generator = get_mel_spectrogram_feature(sample_submission, batch_size=args.batch_size)

    # Load the trained model
    model = load_model(args.model_path)

    # Make predictions
    predictions = predict(model, feature_generator, device)

    # Update the sample submission with predictions
    sample_submission.iloc[:, 1:] = predictions

    # Save the results
    sample_submission.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake voice detection using a CNN model.")
    parser.add_argument('--sample_submission_path', type=str, default='sample_submission.csv',
                        help='Path to the sample submission file')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model file')
    parser.add_argument('--output_path', type=str, default='submission.csv', help='Path to save the submission file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')

    args = parser.parse_args()
    main(args)
