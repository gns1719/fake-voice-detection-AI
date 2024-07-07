# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from model import CRNN
from preprocess import Config
import random
import os
import h5py

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as h5:
            self.num_samples = len(h5['mel'])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5:
            mel = h5['mel'][idx]
            label = h5['label'][idx]
        return torch.FloatTensor(mel).unsqueeze(0), torch.FloatTensor(label)

def load_data():
    train_dataset = H5Dataset('train_data.h5')
    val_dataset = H5Dataset('val_data.h5')
    return train_dataset, val_dataset

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    return np.mean(auc_scores)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

def train(model, optimizer, train_loader, val_loader, device, n_epochs=30):
    criterion = FocalLoss().to(device)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    best_val_score = float('inf')
    best_model = None
    
    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}/train"):
            features, labels = features.float().to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss, val_score = validation(model, criterion, val_loader, device)
        
        scheduler.step()
        
        if val_score < best_val_score:
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
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4)
    
    device = torch.device('cuda')
    
    model = CRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    
    best_model = train(model, optimizer, train_loader, val_loader, device, CONFIG.N_EPOCHS)
    
    torch.save(best_model, 'best_model.pth')
    print("Training completed and best model saved.")