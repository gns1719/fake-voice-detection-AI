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
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
from functools import partial

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    SR = 32000
    N_MELS = 128
    FMAX = 8000
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 32
    N_EPOCHS = 5
    LR = 3e-4
    SEED = 42
    MAX_LEN = 1000
    CHUNK_SIZE = 1000
    INPUT_CHANNELS = 3

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

def get_mel_spectrogram_feature(args):
    path, label, config = args
    try:
        y, sr = librosa.load(path, sr=config.SR)
        
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS, fmax=config.FMAX)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if mel_spectrogram_db.shape[1] > config.MAX_LEN:
            mel_spectrogram_db = mel_spectrogram_db[:, :config.MAX_LEN]
        else:
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, config.MAX_LEN - mel_spectrogram_db.shape[1])), mode='constant')
        
        # Convert to RGB without saving to file
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        plt.tight_layout(pad=0)
        S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=config.FMAX, ax=ax)
        fig.canvas.draw()
        
        rgb_spectrogram = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_spectrogram = rgb_spectrogram.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Resize to match CONFIG.N_MELS and CONFIG.MAX_LEN
        rgb_spectrogram = np.transpose(rgb_spectrogram, (2, 0, 1))  # Change to (channels, height, width)
        rgb_spectrogram = np.array([np.array(Image.fromarray(channel).resize((config.MAX_LEN, config.N_MELS))) for channel in rgb_spectrogram])
        
        label_vector = np.zeros(config.N_CLASSES, dtype=float)
        label_vector[0 if label == 'fake' else 1] = 1
        
        return rgb_spectrogram, label_vector
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None

def process_data_parallel(df, h5_file, dataset_name, mode='w'):
    chunk_size = 1000  # Process 1000 samples at a time

    with h5py.File(h5_file, mode) as f:
        feature_dataset = f.create_dataset(f'{dataset_name}_mel', shape=(len(df), CONFIG.INPUT_CHANNELS, CONFIG.N_MELS, CONFIG.MAX_LEN),
                                           dtype='uint8', chunks=True, maxshape=(None, CONFIG.INPUT_CHANNELS, CONFIG.N_MELS, CONFIG.MAX_LEN))
        label_dataset = f.create_dataset(f'{dataset_name}_labels', shape=(len(df), CONFIG.N_CLASSES),
                                         dtype='float32', chunks=True, maxshape=(None, CONFIG.N_CLASSES))

        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            
            # Prepare arguments for multiprocessing
            args_list = [(row.path, row.label, CONFIG) for _, row in chunk_df.iterrows()]
            
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = list(tqdm(pool.imap(get_mel_spectrogram_feature, args_list), 
                                    total=len(chunk_df), 
                                    desc=f"Processing chunk {i//chunk_size + 1}/{len(df)//chunk_size + 1}"))
            
            valid_results = [r for r in results if r is not None]
            
            if valid_results:
                rgb_spectrograms, label_vectors = zip(*valid_results)
                
                feature_dataset[i:i+len(valid_results)] = np.array(rgb_spectrograms)
                label_dataset[i:i+len(valid_results)] = np.array(label_vectors)
            
            f.flush()  # Ensure data is written to file

if __name__ == "__main__":
    process_data_parallel(train, 'train_data.h5', 'train', mode='w')
    process_data_parallel(val, 'train_data.h5', 'val', mode='a')
    print("Preprocessed data saved successfully.")