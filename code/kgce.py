import torch
import torch.nn as nn
import torch.nn.functional as F

class KGCE(nn.Module):
    def __init__(self, embed_dim=256, num_rubrics=5, tau=0.1, lambda_rubric=1.0):
        super(KGCE, self).__init__()
        self.tau = tau
        self.lambda_rubric = lambda_rubric
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.rubric_classifier = nn.Linear(embed_dim, num_rubrics)

    def forward(self, embedding, rubric_anchors, y_true=None):
        z = self.mlp(embedding)
        # cosine similarity
        sims = F.cosine_similarity(z.unsqueeze(1), rubric_anchors.unsqueeze(0), dim=-1)
        if y_true is not None:
            pos_sims = sims.gather(1, y_true.unsqueeze(1))
            contrastive_loss = -torch.log(pos_sims / sims.sum(dim=1, keepdim=True)).mean()
            rubric_pred = F.softmax(self.rubric_classifier(z), dim=-1)
            rubric_loss = F.cross_entropy(rubric_pred, y_true)
            loss = contrastive_loss + self.lambda_rubric * rubric_loss
            return z, rubric_pred, loss
        else:
            rubric_pred = F.softmax(self.rubric_classifier(z), dim=-1)
            return z, rubric_pred
