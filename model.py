# model.py

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import os
import numpy as np
from preprocess import Config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

CONFIG = Config()

class RNN(nn.Module):
    def __init__(self, input_size=CONFIG.N_MELS, hidden_size=256, num_layers=3, output_dim=CONFIG.N_CLASSES):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.dropout(out[:, -1, :])
        out = F.relu(self.batch_norm(self.fc1(out)))
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    seed_everything(CONFIG.SEED)
    model = RNN()
    save_model(model)