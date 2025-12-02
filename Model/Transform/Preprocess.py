import random

import torch


# Util/Transform/YoloRightPad.py (or wherever you keep transforms)

import torch
import numpy as np


class YoloRightPad(object):
    """
    Pad or truncate YOLO detections to a fixed number of boxes.

    Input:  np.ndarray or torch.Tensor of shape (num_boxes, feat_dim)
            Each row e.g.: [cx, cy, w, h, conf, class_id_norm]
    Output: torch.Tensor of shape (feat_dim, width)
    """

    def __init__(self, width: int):
        self.width = width

    def __call__(self, sample):
        # sample: np.ndarray or torch.Tensor (num_boxes, feat_dim)
        if isinstance(sample, torch.Tensor):
            arr = sample.cpu().numpy()
        else:
            arr = np.asarray(sample, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]  # (feat_dim,) -> (1, feat_dim)

        num_boxes, feat_dim = arr.shape

        if num_boxes > 0:
            # sort by confidence descending (assuming conf is column 4)
            # adjust index if your feature layout differs
            conf_col = 4
            order = np.argsort(-arr[:, conf_col])
            arr = arr[order]

        # truncate or pad
        if num_boxes >= self.width:
            arr = arr[: self.width, :]
        else:
            pad_rows = self.width - num_boxes
            pad = np.zeros((pad_rows, feat_dim), dtype=np.float32)
            arr = np.vstack([arr, pad])

        # shape: (width, feat_dim) -> (feat_dim, width)
        arr = arr.T  # (feat_dim, width)

        return torch.from_numpy(arr).float()
