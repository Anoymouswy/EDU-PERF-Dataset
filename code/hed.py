import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class HED(nn.Module):
    def __init__(self, audio_dim, visual_dim, embed_dim=256, n_heads=4, n_layers=2):
        super(HED, self).__init__()
        self.input_proj = nn.Linear(audio_dim+visual_dim, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, embed_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, audio, visual):
        x = torch.cat([audio, visual], dim=-1)
        x = self.input_proj(x)
        h, _ = self.bilstm(x)
        # mean pooling per segment; here use full sequence as demo
        s = h.mean(dim=1, keepdim=True)
        h_global = self.transformer(s)
        y_exp = self.regressor(h_global.mean(dim=1))
        return y_exp
