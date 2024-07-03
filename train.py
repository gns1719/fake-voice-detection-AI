# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from model import MLP, Config
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

def load_data():
    train_mel = pd.read_csv('train_mel.csv').values
    train_labels = pd.read_csv('train_labels.csv').values
    val_mel = pd.read_csv('val_mel.csv').values
    val_labels = pd.read_csv('val_labels.csv').values
    
    train_dataset = CustomDataset(train_mel, train_labels)
    val_dataset = CustomDataset(val_mel, val_labels)
    
    return train_dataset, val_dataset

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    return np.mean(auc_scores)

def train(model, optimizer, train_loader, val_loader, device, n_epochs=5):
    criterion = nn.BCELoss().to(device)
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}/train"):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss, val_score = validation(model, criterion, val_loader, device)
        train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss: [{train_loss:.5f}] Val Loss: [{val_loss:.5f}] Val AUC: [{val_score:.5f}]')
        
        if best_val_score < val_score:
            best_val_score = val_score
            best_model = model.state_dict()
    
    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            loss = criterion(probs, labels)
            
            val_loss.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    val_loss = np.mean(val_loss)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return val_loss, auc_score

if __name__ == "__main__":
    CONFIG = Config()
    seed_everything(CONFIG.SEED)
    
    device = torch.device('cuda')
    
    train_dataset, val_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False)
    
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    best_model = train(model, optimizer, train_loader, val_loader, device)
    
    torch.save(best_model, 'best_model.pth')
    print("Training completed. Best model saved to 'best_model.pth'")
