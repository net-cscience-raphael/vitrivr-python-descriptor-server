# Model/Dataset/YoloClassification.py

import csv
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class Mode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class Split(Enum):
    TRAIN = 1
    TRAIN_VALIDATION = 2
    TRAIN_VALIDATION_TEST = 3


class Modal(Enum):
    CLASS_YOLOCLASSIFICATION = 1


# ---------------------------------------------------------------------
# Feature configuration / names
# ---------------------------------------------------------------------

# thresholds for small / mid / large in terms of area fraction
SMALL_THRESH = 0.01     # < 1% of image
LARGE_THRESH = 0.20     # > 20% of image

FEATURE_NAMES = [
    "num_objects",          # 0
    "num_unique_classes",   # 1
    "mean_conf",            # 2
    "max_conf",             # 3
    "sum_conf",             # 4
    "mean_area_frac",       # 5
    "max_area_frac",        # 6
    "total_area_frac",      # 7
    "num_large_objects",    # 8
    "num_small_objects",    # 9
    "spread_x",             # 10
    "spread_y",             # 11
    "grid_occupancy_3x3",   # 12
    "num_mid_objects",      # 13
    "mean_area_small",      # 14
    "mean_area_mid",        # 15
    "mean_area_large",      # 16
    "mean_aspect_ratio",    # 17
    "std_aspect_ratio",     # 18
    "skew_x",               # 19
    "kurt_x",               # 20
    "skew_y",               # 21
    "kurt_y",               # 22
]


# ---------------------------------------------------------------------
# Helper: 3rd & 4th moments (skewness, kurtosis)
# ---------------------------------------------------------------------

def _skew_kurtosis(arr: np.ndarray) -> (float, float):
    """
    Compute skewness and excess kurtosis of a 1D array.
    Returns (skew, kurtosis), both 0.0 if not defined or variance ~ 0.
    """
    if arr.size == 0:
        return 0.0, 0.0

    mean = arr.mean()
    diff = arr - mean
    m2 = np.mean(diff ** 2)

    if m2 <= 1e-12:
        # Almost no variance -> distribution is basically a spike
        return 0.0, 0.0

    m3 = np.mean(diff ** 3)
    m4 = np.mean(diff ** 4)

    # standardized skewness and excess kurtosis
    skew = m3 / (m2 ** 1.5 + 1e-12)
    kurt = m4 / (m2 ** 2 + 1e-12) - 3.0
    return float(skew), float(kurt)


def extract_features_from_response(resp: Dict[str, Any]) -> np.ndarray:
    """
    Given the JSON response (or precomputed JSON) in the format:
        {
            "image": {"width": W, "height": H, ...},
            "detections": [
                {
                  "bbox": {"x1":..., "y1":..., "x2":..., "y2":...},
                  "confidence": float,
                  "class_id": int (optional),
                  "class_name": str (optional)
                },
                ...
            ]
        }

    compute a statistical feature vector (no raw positions as direct features).
    """
    detections = resp.get("detections", [])
    img_info = resp.get("image", {})
    width = float(img_info.get("width", 1.0))
    height = float(img_info.get("height", 1.0))
    image_area = max(width * height, 1.0)

    num_objects = len(detections)
    if num_objects == 0:
        # Return zeros for all features
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    confidences = []
    areas_frac = []
    class_ids = set()

    centers_x_norm = []
    centers_y_norm = []
    aspect_ratios = []

    for det in detections:
        conf = float(det.get("confidence", 0.0))
        confidences.append(conf)

        bbox = det.get("bbox", {})
        x1 = float(bbox.get("x1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        w_box = max(x2 - x1, 0.0)
        h_box = max(y2 - y1, 0.0)
        box_area_frac = (w_box * h_box) / image_area
        areas_frac.append(box_area_frac)

        # Center in normalized [0,1] coordinates
        cx_norm = ((x1 + x2) * 0.5) / max(width, 1e-6)
        cy_norm = ((y1 + y2) * 0.5) / max(height, 1e-6)
        centers_x_norm.append(cx_norm)
        centers_y_norm.append(cy_norm)

        # Aspect ratio (w/h)
        aspect = w_box / (h_box + 1e-6)
        aspect_ratios.append(aspect)

        # we only use class_id to count how many unique classes, not the identity
        cid = det.get("class_id", None)
        if cid is not None:
            class_ids.add(cid)

    confidences = np.array(confidences, dtype=float)
    areas_frac = np.array(areas_frac, dtype=float)
    centers_x_norm = np.array(centers_x_norm, dtype=float)
    centers_y_norm = np.array(centers_y_norm, dtype=float)
    aspect_ratios = np.array(aspect_ratios, dtype=float)

    # Size category masks
    small_mask = areas_frac < SMALL_THRESH
    mid_mask = (areas_frac >= SMALL_THRESH) & (areas_frac <= LARGE_THRESH)
    large_mask = areas_frac > LARGE_THRESH

    num_small = int(small_mask.sum())
    num_mid = int(mid_mask.sum())
    num_large = int(large_mask.sum())

    # Per-size mean areas (0 if no objects in that category)
    mean_area_small = float(areas_frac[small_mask].mean()) if num_small > 0 else 0.0
    mean_area_mid = float(areas_frac[mid_mask].mean()) if num_mid > 0 else 0.0
    mean_area_large = float(areas_frac[large_mask].mean()) if num_large > 0 else 0.0

    # Spatial distribution: std of box centers
    spread_x = float(centers_x_norm.std())  # horizontal spread
    spread_y = float(centers_y_norm.std())  # vertical spread

    # 3x3 grid occupancy of centers
    cell_x = np.clip((centers_x_norm * 3).astype(int), 0, 2)
    cell_y = np.clip((centers_y_norm * 3).astype(int), 0, 2)
    unique_cells = set(zip(cell_x.tolist(), cell_y.tolist()))
    grid_occupancy = float(len(unique_cells)) / 9.0

    # Aspect ratio stats
    mean_aspect_ratio = float(aspect_ratios.mean())
    std_aspect_ratio = float(aspect_ratios.std())

    # 3rd & 4th moments (as skewness + kurtosis) of spatial distribution
    skew_x, kurt_x = _skew_kurtosis(centers_x_norm)
    skew_y, kurt_y = _skew_kurtosis(centers_y_norm)

    features = np.array([
        float(num_objects),                  # num_objects
        float(len(class_ids)),               # num_unique_classes
        float(confidences.mean()),           # mean_conf
        float(confidences.max()),            # max_conf
        float(confidences.sum()),            # sum_conf
        float(areas_frac.mean()),            # mean_area_frac
        float(areas_frac.max()),             # max_area_frac
        float(areas_frac.sum()),             # total_area_frac
        float(num_large),                    # num_large_objects
        float(num_small),                    # num_small_objects

        spread_x,                            # spread_x
        spread_y,                            # spread_y
        grid_occupancy,                      # grid_occupancy_3x3

        float(num_mid),                      # num_mid_objects
        mean_area_small,                     # mean_area_small
        mean_area_mid,                       # mean_area_mid
        mean_area_large,                     # mean_area_large

        mean_aspect_ratio,                   # mean_aspect_ratio
        std_aspect_ratio,                    # std_aspect_ratio

        skew_x,                              # skew_x (centers_x_norm)
        kurt_x,                              # kurt_x (centers_x_norm)
        skew_y,                              # skew_y (centers_y_norm)
        kurt_y,                              # kurt_y (centers_y_norm)
    ], dtype=float)

    return features


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class YoloClassification(Dataset):
    """
    Dataset that:
      - reads (image_path, label, split) from a CSV
      - loads YOLO(-World) detections from precomputed JSON files
      - extracts a global statistical feature vector per image
      - returns (X, y) where X.shape = (len(FEATURE_NAMES),)
    """

    def __init__(
        self,
        root: str,
        mode: Mode,
        split_strategy: Split,
        json_root: str = "yolo_json",
        transform=None,
        ):
        super().__init__()

        self.root = Path(root).expanduser().resolve()
        #Do it batch wise!!!
        #self.device = device
        self.mode = mode
        self.split_strategy = split_strategy
        self.transform = transform

        jr = Path(json_root)
        self.json_root = jr if jr.is_absolute() else (self.root / jr)

        self.samples = self._load_metadata()

    # -------------------- split handling --------------------

    def _load_metadata(self):
        """
        Expect a CSV file: root/annotations.csv with fields:
            image_path,label,split

        where split ∈ {"train", "val", "validation", "test"}.
        """
        meta_path = self.root / "annotations.csv"
        samples = []
        with meta_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                label = int(row["label"])
                split = row.get("split", "train").lower()

                if self._is_in_split(split):
                    samples.append((img_path, label))
        return samples

    def _is_in_split(self, split: str) -> bool:
        if self.mode == Mode.TRAIN:
            return split == "train"
        elif self.mode == Mode.VALIDATION:
            return split in {"val", "validation"}
        elif self.mode == Mode.TEST:
            return split == "test"
        return False

    def __len__(self):
        return len(self.samples)

    # -------------------- JSON access --------------------

    def _load_resp_from_json(self, img_rel_path: str) -> Dict[str, Any]:
        """
        Load full response dict from precomputed JSON:
          json_root / (image_path + ".json")

        Returns dict with at least {"detections": [...], "image": {...}}
        or {"detections": [], "image": {}} if missing / error.
        """
        json_path = self.json_root / f"{img_rel_path}.json"
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Make sure keys exist
            if "detections" not in data:
                data["detections"] = []
            if "image" not in data:
                data["image"] = {}
            return data
        except FileNotFoundError:
            print(f"⚠️ JSON not found for {img_rel_path}: {json_path}")
            return {"detections": [], "image": {}}
        except Exception as e:
            print(f"⚠️ Failed to read JSON {json_path}: {e}")
            return {"detections": [], "image": {}}

    # -------------------- main access --------------------

    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]

        # if you ever want full image path:
        # full_path = self.root / img_rel_path

        # 1) load YOLO response from JSON
        resp = self._load_resp_from_json(img_rel_path)

        # 2) extract statistical feature vector
        feat_vec = extract_features_from_response(resp)  # np.ndarray, shape (F,)

        # 3) convert to torch tensor
        X = torch.from_numpy(feat_vec.astype(np.float32))

        # 4) label tensor (binary classification 0/1)
        y = torch.tensor(float(label), dtype=torch.float32)

        return X, y
