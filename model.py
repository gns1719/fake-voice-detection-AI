# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from preprocess import Config

CONFIG = Config()

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes=CONFIG.N_CLASSES):
        super(CNNWithAttention, self).__init__()
        
        # Use a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace the first layer to accept 3-channel input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    model = CNNWithAttention()
    save_model(model)