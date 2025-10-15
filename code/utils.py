import torch

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for audio, visual, score, label_align, label_expr in dataloader:
        audio, visual, score = audio.to(device), visual.to(device), score.to(device)
        label_align, label_expr = label_align.to(device), label_expr.to(device)
        optimizer.zero_grad()
        align_pred, expr_score, _, _, kgce_loss = model(audio, visual, score, rubric_anchors=torch.rand(5,256).to(device), y_rubric=torch.randint(0,5,(audio.size(0),)).to(device))
        loss_align = nn.CrossEntropyLoss()(align_pred.view(-1,5), label_align.view(-1))
        loss_expr = nn.MSELoss()(expr_score.view(-1), label_expr.view(-1))
        loss = loss_align + loss_expr + kgce_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for audio, visual, score, label_align, label_expr in dataloader:
            audio, visual, score = audio.to(device), visual.to(device), score.to(device)
            label_align, label_expr = label_align.to(device), label_expr.to(device)
            align_pred, expr_score, _, _, kgce_loss = model(audio, visual, score, rubric_anchors=torch.rand(5,256).to(device), y_rubric=torch.randint(0,5,(audio.size(0),)).to(device))
            loss_align = nn.CrossEntropyLoss()(align_pred.view(-1,5), label_align.view(-1))
            loss_expr = nn.MSELoss()(expr_score.view(-1), label_expr.view(-1))
            loss = loss_align + loss_expr + kgce_loss
            total_loss += loss.item()
    return total_loss/len(dataloader)
