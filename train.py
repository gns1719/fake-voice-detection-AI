# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import Config
from model import get_model
from tqdm import tqdm
import os

class AudioDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        melspec = np.load(f"{self.data_dir}/{row['id']}.npy")
        label = 1 if row['label'] == 'real' else 0
        return torch.FloatTensor(melspec), torch.LongTensor([label])

def check_data_files(df, data_dir):
    missing_files = []
    for _, row in df.iterrows():
        file_path = os.path.join(data_dir, f"{row['id']}.npy")
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    return missing_files

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target.squeeze()).sum().item()
            
            progress_bar.set_postfix({
                'train_loss': f"{train_loss/(batch_idx+1):.4f}",
                'train_acc': f"{100.*correct/total:.2f}%"
            })

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target.squeeze()).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target.squeeze()).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train Accuracy: {100.*correct/total:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {100.*correct/total:.2f}%")

if __name__ == "__main__":
    try:
        print("Starting script execution")
        
        print("Loading and splitting data")
        df = pd.read_csv("train.csv")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=Config.SEED)
        
        print("Checking for missing data files")
        missing_train = check_data_files(train_df, "processed_train")
        missing_val = check_data_files(val_df, "processed_train")
        
        if missing_train or missing_val:
            print(f"Missing train files: {len(missing_train)}")
            print(f"Missing validation files: {len(missing_val)}")
            raise FileNotFoundError("Some preprocessed data files are missing.")
        
        print("Creating datasets")
        train_dataset = AudioDataset(train_df, "processed_train")
        val_dataset = AudioDataset(val_df, "processed_train")
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        print("Creating data loaders")
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        print("Checking CUDA availability")
        device = torch.device("cuda")
        print(f"Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()}")
            print(f"GPU memory cached: {torch.cuda.memory_cached()}")
        
        print("Creating model")
        model = get_model(use_attention=True).to(device)
        print(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)
        
        print("Starting training")
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=Config.N_EPOCHS, device=device)
        
        print("Saving model")
        torch.save(model.state_dict(), "deep_fake_voice_model.pth")
        
        print("Training completed successfully")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()