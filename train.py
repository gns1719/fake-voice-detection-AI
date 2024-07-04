# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from model import CNN
from preprocess import Config
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

class CustomDataset(Dataset):
    def __init__(self, mel, label):
        self.mel = mel
        self.label = label

    def __len__(self):
        return len(self.mel)

    def __getitem__(self, idx):
        if self.label is not None:
            return torch.FloatTensor(self.mel[idx]).unsqueeze(0), torch.FloatTensor(self.label[idx])
        return torch.FloatTensor(self.mel[idx]).unsqueeze(0)

def load_data():
    train_mel = np.load('train_mel.npy')
    train_labels = np.load('train_labels.npy')
    val_mel = np.load('val_mel.npy')
    val_labels = np.load('val_labels.npy')
    
    train_dataset = CustomDataset(train_mel, train_labels)
    val_dataset = CustomDataset(val_mel, val_labels)
    
    return train_dataset, val_dataset

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
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
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
            features = features.float().to(device)
            labels = labels.float().to(device)
            
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
    
    train_dataset, val_dataset = load_data()
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    
    best_model = train(model, optimizer, train_loader, val_loader, device, CONFIG.N_EPOCHS)
    
    torch.save(best_model, 'best_model.pth')
    print("Training completed and best model saved.")