
import torch
from torch import nn

from Model.Dataset.YoloClassification import FEATURE_NAMES
from Model.Transform.Preprocess import YoloRightPad  # adjust import path if needed


class ViewpointDense_2(nn.Module):
    def __init__(self, max_boxes: int = 1):
        super().__init__()
        self.max_boxes = max_boxes
        self.feat_dim = 5
        self.dropout = 0.2  # a bit more dropout

        input_dim = len(FEATURE_NAMES) * self.max_boxes

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(32, 1),  # logits
        )

    def forward(self, x):
        x_f = self.flatten(x)
        logits = self.net(x_f)
        return logits.squeeze(-1)

    class CustomLoss(nn.Module):
        def __init__(self, pos_weight=None):
            super().__init__()
            if pos_weight is not None:
                self.bce = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight])
                )
            else:
                self.bce = nn.BCEWithLogitsLoss()

        def forward(self, y_hat, y, **kwargs):
            return self.bce(y_hat, y)



    def get_loss(self, device, pos_weight: float = 1.0):
        return self.CustomLoss(pos_weight=pos_weight).to(device)


    def get_optimizer(self, model, r_learning):
        optimizer = torch.optim.Adam(model.parameters(), lr=r_learning, weight_decay=r_learning/10)
        return optimizer

    def get_transforms(self, device):
        return