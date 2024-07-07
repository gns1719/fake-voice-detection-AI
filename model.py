# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import Config

CONFIG = Config()

class CRNN(nn.Module):
    def __init__(self, num_classes=CONFIG.N_CLASSES):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the input size for the GRU layer
        self.gru_input_size = 256 * 125  # 125 is the size after the last pooling layer
        
        # RNN layers
        self.gru = nn.GRU(self.gru_input_size, 128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Prepare for RNN
        batch, channels, height, width = x.size()
        x = x.view(batch, height, channels * width)
        
        # RNN layers
        x, _ = self.gru(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return torch.sigmoid(x)

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    model = CRNN()
    save_model(model)