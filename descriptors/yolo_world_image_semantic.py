import base64
import importlib
import logging
from io import BytesIO

import torch
from PIL import Image
from apiflask import APIBlueprint
from flask import request
import numpy as np

from Model.Dataset.YoloClassification import extract_features_from_response
from descriptors.image_cache.ImageCache import ImageCache
from descriptors.yolo_world_image import yolo_image_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Yolo is using device: {device}")

yolo_world_image_semantic_bp = APIBlueprint('yolo_world_image_semantic', __name__)

image_cache = ImageCache()

@yolo_world_image_semantic_bp.post("/extract/yolo_world_image_semantic")
@yolo_world_image_semantic_bp.doc(
    summary="Yolo world endpoint for feature extraction on image, where the image is transmitted in the body by a data URL")
def yolo_world_image_semantic():
    logging.info(f"Embedding image request received on YOLO-World endpoint with device {device}")

    data = request.form.get('data', '')
    header, encoded = data.split("base64,", 1)
    conf_level = float( request.form.get('confLevel', '0.25'))
    image_size = int( request.form.get('imageSize', 640))
    persist= bool( request.form.get('persist', False))
    do_cache = bool(request.form.get('cache', False))
    image_id = str(request.form.get('image_id'))
    # Parse a comma seperated filter to a  list if provided to remove words from model
    filters = request.form.get('filters', None)

    image = Image.open(BytesIO(base64.b64decode(encoded)))

    if do_cache and image_id:
        image_cache.put(image, key=image_id)
    resp = yolo_image_inference(image, conf_level, image_size, filters, persist)
    model = load_model(
        model_name="Dense.ViewpointDense_2",
        weights_path="./local-models/exp_ViewpointDense_2_e100.pth",
        device="cuda"
    )
    X = torch.from_numpy(extract_features_from_response(resp).astype(np.float32))
    with torch.no_grad():
        features = model(torch.unsqueeze(X.to(device),0)).sigmoid().cpu().numpy().flatten().tolist()
    return features



@yolo_world_image_semantic_bp.post("/extract/yolo_world_image_semantic_from_cache")
@yolo_world_image_semantic_bp.doc(
    summary="Yolo world endpoint for feature extraction on image, where the image is transmitted in the body by a data URL")
def yolo_world_image_semantic_from_cache():
    image_id = str(request.form.get('image_id'))
    conf_level = float(request.form.get('confLevel', '0.25'))
    image_size = int(request.form.get('imageSize', 640))
    persist = bool(request.form.get('persist', False))
    # Parse a comma seperated filter to a  list if provided to remove words from model
    filters = request.form.get('filters', None)

    image = image_cache.get(image_id)
    image = image.copy()
    resp = yolo_image_inference(image, conf_level, image_size, filters, persist)
    model = load_model(
        model_name="Dense.ViewpointDense_2",
        weights_path="./local-models/exp_ViewpointDense_2_e100.pth",
        device="cuda"
    )
    X = torch.from_numpy(extract_features_from_response(resp).astype(np.float32))
    with torch.no_grad():
        features = model(torch.unsqueeze(X.to(device),0)).sigmoid().cpu().numpy().flatten().tolist()
    return features



def load_model(model_name: str, weights_path: str, device="cpu"):
    # dynamic import: Model.Models.<model_name>
    module = importlib.import_module(f"Model.Models.{model_name}")
    class_name = model_name.split(".")[1]   # e.g. "ViewpointDense_2"
    clazz = getattr(module, class_name)

    # instantiate model
    model = clazz().to(device)

    # load state dict
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    model.eval()
    return model

