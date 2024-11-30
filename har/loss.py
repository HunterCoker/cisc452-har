from torch.nn.functional import log_softmax
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        log_probabilities = log_softmax(inputs, dim=1)
        probabilities = torch.exp(log_probabilities)
        focal_loss = -(1 - probabilities) ** self.gamma * log_probabilities
        if self.alpha is not None:
            focal_loss *= self.alpha
        return focal_loss.gather(1, targets.unsqueeze(1)).mean()