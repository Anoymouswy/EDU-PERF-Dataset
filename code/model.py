import torch
import torch.nn as nn
from .caam import CAAM
from .hed import HED
from .kgce import KGCE

class AutoMPEModel(nn.Module):
    def __init__(self, score_dim=3, audio_dim=128, visual_dim=17, embed_dim=256):
        super(AutoMPEModel, self).__init__()
        self.caam = CAAM(score_dim, audio_dim, embed_dim)
        self.hed = HED(audio_dim, visual_dim, embed_dim)
        self.kgce = KGCE(embed_dim)

    def forward(self, audio, visual, score, rubric_anchors=None, y_rubric=None):
        E_CAAM, align_pred = self.caam(score, audio)
        expr_score = self.hed(audio, visual)
        if rubric_anchors is not None:
            embedding, rubric_pred, kgce_loss = self.kgce(E_CAAM, rubric_anchors, y_rubric)
            return align_pred, expr_score, embedding, rubric_pred, kgce_loss
        else:
            embedding, rubric_pred = self.kgce(E_CAAM, rubric_anchors)
            return align_pred, expr_score, embedding, rubric_pred
