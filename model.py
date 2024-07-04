# model.py

import torch
import torch.nn as nn
import random
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

class CNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=CONFIG.N_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Calculate the size of the flattened features
        self.flat_features = 128 * (CONFIG.N_MELS // 8) * (CONFIG.MAX_LEN // 8)

        self.fc1 = nn.Linear(self.flat_features, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    seed_everything(CONFIG.SEED)
    model = CNN()
    save_model(model)