import torch
import torch.nn as nn
import torch.nn.functional as F

class CAAM(nn.Module):
    def __init__(self, score_dim, audio_dim, embed_dim=256):
        super(CAAM, self).__init__()
        self.embed_dim = embed_dim
        self.W_Qs = nn.Linear(score_dim, embed_dim)
        self.W_Ka = nn.Linear(audio_dim, embed_dim)
        self.W_Va = nn.Linear(audio_dim, embed_dim)
        
        self.W_Qa = nn.Linear(audio_dim, embed_dim)
        self.W_Ks = nn.Linear(score_dim, embed_dim)
        self.W_Vs = nn.Linear(score_dim, embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 5)

    def forward(self, score, audio):
        # score -> audio
        Qs = self.W_Qs(score)
        Ka = self.W_Ka(audio)
        Va = self.W_Va(audio)
        Hs2a = F.softmax(Qs @ Ka.transpose(-2,-1)/self.embed_dim**0.5, dim=-1) @ Va
        
        # audio -> score
        Qa = self.W_Qa(audio)
        Ks = self.W_Ks(score)
        Vs = self.W_Vs(score)
        Ha2s = F.softmax(Qa @ Ks.transpose(-2,-1)/self.embed_dim**0.5, dim=-1) @ Vs
        
        H = torch.cat([Hs2a, Ha2s], dim=-1)
        E_CAAM = self.ffn(H)
        alignment_pred = self.classifier(E_CAAM)
        return E_CAAM, alignment_pred
