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
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Add new fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ResNet features
        features = self.resnet(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Global average pooling
        features = torch.mean(features.view(features.size(0), features.size(1), -1), dim=2)
        
        # Fully connected layers
        output = self.fc(features)
        
        return output

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    model = CNNWithAttention()
    save_model(model)