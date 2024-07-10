# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import Config

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        context_vector = torch.sum(x * attention_weights, dim=1)
        return context_vector

class DetailedRNN(nn.Module):
    def __init__(self, input_size=Config.N_MELS, hidden_size=256, num_layers=4, output_dim=Config.N_CLASSES, use_attention=True):
        super(DetailedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        
        if use_attention:
            self.attention = AttentionLayer(hidden_size * 2)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_dim)
        
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        
        if self.use_attention:
            out = self.attention(out)
        else:
            out = out[:, -1, :]
        
        out = self.dropout(out)
        out = F.relu(self.batch_norm1(self.fc1(out)))
        out = self.dropout(out)
        out = F.relu(self.batch_norm2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

def get_model(use_attention=True):
    return DetailedRNN(use_attention=use_attention)