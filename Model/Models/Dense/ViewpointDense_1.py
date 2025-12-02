# Model/Models/ViewpointDense_1.py

import torch
from torch import nn

from Model.Dataset.YoloClassification import FEATURE_NAMES
from Model.Transform.Preprocess import YoloRightPad  # adjust import path if needed


class ViewpointDense_1(nn.Module):
    """
    Dense classifier on padded YOLO detections.

    Input shape per sample: (feat_dim, width)
    Example: feat_dim=6, width=120 -> flattened to 6*120 = 720
    """

    def __init__(self, max_boxes: int = 1):
        super().__init__()
        self.max_boxes = max_boxes
        self.feat_dim = 5
        self.dropout = 0.1

        input_dim = len(FEATURE_NAMES) * self.max_boxes  # 5 * 120

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1),  # binary classification: OK/NOK
        )

    def forward(self, x):
        # x: (batch, 5, max_boxes)
        x_f = self.flatten(x)
        logits = self.net(x_f)
        return logits.squeeze(-1)  # (batch,)



    class CustomLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, y_hat, y, epoch=None, epochs=None, wandb=None):
            # y_hat: (batch,) logits
            # y:     (batch,) float {0,1}
            return self.bce(y_hat, y)

    def get_loss(self, device):
        return self.CustomLoss().to(device)


    def get_optimizer(self, model, r_learning):
        optimizer = torch.optim.Adam(model.parameters(), lr=r_learning, weight_decay=r_learning/10)
        return optimizer

    def get_transforms(self, device):
        return YoloRightPad(width=self.max_boxes)