import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MusicPerformanceDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        audio = torch.tensor(data['audio'], dtype=torch.float32)
        visual = torch.tensor(data['visual'], dtype=torch.float32)
        score = torch.tensor(data['score'], dtype=torch.float32)
        label_alignment = torch.tensor(data['label_alignment'], dtype=torch.long)
        label_expression = torch.tensor(data['label_expression'], dtype=torch.float32)
        return audio, visual, score, label_alignment, label_expression
