import torch
from torch.utils.data import DataLoader
from auto_mpe import AutoMPEModel, MusicPerformanceDataset, evaluate_model
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MusicPerformanceDataset("data/demo_data")
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

model = AutoMPEModel()
os.makedirs("checkpoints", exist_ok=True)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)

loss = evaluate_model(model, dataloader, device)
print(f"Evaluation Loss: {loss:.4f}")
