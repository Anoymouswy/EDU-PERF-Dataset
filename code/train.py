import torch
from torch.utils.data import DataLoader
from auto_mpe import AutoMPEModel, MusicPerformanceDataset, train_one_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MusicPerformanceDataset("data/demo_data")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = AutoMPEModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    loss = train_one_epoch(model, dataloader, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

torch.save(model.state_dict(), "checkpoints/best_model.pth")
