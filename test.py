# test.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from preprocess import Config
from model import get_model

class TestDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        melspec = np.load(f"{self.data_dir}/{row['id']}.npy")
        return torch.FloatTensor(melspec), row['id']

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    ids = []
    with torch.no_grad():
        for data, file_id in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            predictions.extend(probs.cpu().numpy())
            ids.extend(file_id)
    return ids, predictions

if __name__ == "__main__":
    test_df = pd.read_csv("test.csv")
    test_dataset = TestDataset(test_df, "processed_test")
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(use_attention=True).to(device)
    model.load_state_dict(torch.load("deep_fake_voice_model.pth"))

    ids, predictions = predict(model, test_loader, device)

    submission = pd.DataFrame({
        "id": ids,
        "fake": [pred[0] for pred in predictions],
        "real": [pred[1] for pred in predictions]
    })

    submission.to_csv("sample_submission.csv", index=False)
    print("Predictions saved to sample_submission.csv")