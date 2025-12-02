import torch
from torch import nn

class CustomLoss(nn.Module):


    def __init__(self, loss_weights: dict):
        super().__init__()

    def forward(self, y_hat, y):
        return