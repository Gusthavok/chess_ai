import torch
import torch.nn as nn

class AdaptiveMSELoss(nn.Module):
    def __init__(self, lambda_factor=1.0):
        super(AdaptiveMSELoss, self).__init__()
        self.lambda_factor = lambda_factor

    def forward(self, y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)  # Erreur quadratique moyenne (MSE)
        adaptive_factor = 1 + self.lambda_factor * torch.abs(y_true)  # Facteur adaptatif
        loss = mse / adaptive_factor  # Application du facteur adaptatif
        return torch.sum(loss)
