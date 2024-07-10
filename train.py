# train.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import preprocess
from model import CNN
from preprocess import Config


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


#TODO 점검해볼것
def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    return np.mean(auc_scores)


def train(model, optimizer, criterion, train_loader, val_loader, device, n_epochs):
    best_val_score = 0
    best_model = None
    patience = 10
    no_improve = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

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

        train_loss = np.mean(train_loss)

        # 검증 (점수)
        val_loss, val_auc, val_accuracy, val_f1 = validation(model, criterion, val_loader, device)

        # 학습률 조정
        scheduler.step(val_loss)

        print(f'Epoch [{epoch}], Train Loss: [{train_loss:.5f}], '
              f'Val Loss: [{val_loss:.5f}], Val AUC: [{val_auc:.5f}], '
              f'Val Accuracy: [{val_accuracy:.5f}], Val F1: [{val_f1:.5f}]')

        # 최고 모델 저장 및 조기 종료 체크
        #TODO
        # auc 점수로 모델의 정확도 체크
        # 다른 점수들도 사용 고려
        if val_auc > best_val_score:
            best_val_score = val_auc
            best_model = model.state_dict()
            torch.save(best_model, f'checkpoint_epoch_{epoch}.pth')
            print(f"New best model saved with AUC: {best_val_score:.5f}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
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

    # 추가 메트릭 계산
    auc_score = multiLabel_AUC(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_probs > 0.5).astype(int), average='micro')

    print(f"Validation - Loss: {val_loss:.4f}, AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return val_loss, auc_score, accuracy, f1


if __name__ == "__main__":
    CONFIG = Config()
    preprocess.seed_everything(CONFIG.SEED)
    
    device = torch.device('cuda')
    
    train_dataset, val_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False)

    model = CNN(n_classes=CONFIG.N_CLASSES).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)

    best_model = train(model, optimizer, criterion, train_loader, val_loader, device, n_epochs=CONFIG.N_EPOCHS)

    torch.save(best_model, 'best_model.pth')
    print("Training completed. Best model saved to 'best_model.pth'")
