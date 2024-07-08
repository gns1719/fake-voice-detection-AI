import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from model import CNNWithAttention
from preprocess import Config, seed_everything
import random
import os
import h5py

CONFIG = Config()

class CustomDataset(Dataset):
    def __init__(self, h5_file, dataset_name, test_mode=False):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.test_mode = test_mode
        
        with h5py.File(self.h5_file, 'r') as f:
            self.data_len = f[dataset_name + '_mel'].shape[0]
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            feature = torch.tensor(f[self.dataset_name + '_mel'][idx], dtype=torch.float32)
            if not self.test_mode:
                label = torch.tensor(f[self.dataset_name + '_labels'][idx], dtype=torch.float32)
                return feature, label
            else:
                return feature

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    return np.mean(auc_scores)

def train(model, optimizer, train_loader, val_loader, device, n_epochs=10):
    criterion = nn.BCELoss().to(device)
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}/train"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss, val_score = validation(model, criterion, val_loader, device)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model.state_dict()
        
        print(f"Epoch {epoch}: Train Loss: {np.mean(train_loss):.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_score:.4f}")

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_true = []
    val_pred = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)
            
            output = model(features)
            loss = criterion(output, labels)
            val_loss.append(loss.item())
            
            val_true.append(labels.cpu().numpy())
            val_pred.append(output.cpu().numpy())
    
    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)
    
    val_score = multiLabel_AUC(val_true, val_pred)
    
    return np.mean(val_loss), val_score

if __name__ == "__main__":
    seed_everything(CONFIG.SEED)
    
    train_dataset = CustomDataset('train_data.h5', 'train')
    val_dataset = CustomDataset('train_data.h5', 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNNWithAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    
    best_model = train(model, optimizer, train_loader, val_loader, device, CONFIG.N_EPOCHS)
    
    torch.save(best_model, 'best_model.pth')
    print("Training completed and best model saved.")