import base64
from io import BytesIO
from typing import Any, Dict, List
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from apiflask import APIBlueprint
from flask import request
import uuid
import logging

from descriptors.image_cache.ImageCache import ImageCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Yolo is using device: {device}")

yolo_world_image = APIBlueprint('yolo_world_image', __name__)

model_name = "yolov8x-worldv2"

try:
    from ultralytics import YOLO
    from ultralytics import YOLOWorld
except ImportError as e:
    logging.error("Ultralytics YOLO package is not installed. Please install it with 'pip install ultralytics'.")
    raise


classes_list = []

with open("descriptors/yolo_world_objects_2k.txt", "r", encoding="utf-8") as f:
    classes_list = [line.strip() for line in f if line.strip()]


try:
    model = YOLOWorld(f"./models_cache/{model_name}")

    if classes_list:
        try:
            model.set_classes(classes_list)
        except Exception as e:
            logging.warn( f"Failed to set classes on YOLO-World model: {e}")

except Exception as e:
    logging.error( f" Failed to load model '{model_name}': {e}")


image_cache = ImageCache()

@yolo_world_image.post("/extract/yolo_world_image")
@yolo_world_image.doc(
    summary="Yolo world endpoint for feature extraction on image, where the image is transmitted in the body by a data URL")
def yolo_image():
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

    return yolo_image_inference(image, conf_level, image_size, filters, persist)


@yolo_world_image.post("/extract/yolo_world_image_from_cache")
@yolo_world_image.doc(
    summary="Yolo world endpoint for feature extraction on image, where the image is transmitted in the body by a data URL")
def yolo_image_from_cache():
    image_id = str(request.form.get('image_id'))
    conf_level = float(request.form.get('confLevel', '0.25'))
    image_size = int(request.form.get('imageSize', 640))
    persist = bool(request.form.get('persist', False))
    # Parse a comma seperated filter to a  list if provided to remove words from model
    filters = request.form.get('filters', None)

    image = image_cache.get(image_id)
    image = image.copy()

    return yolo_image_inference(image, conf_level, image_size, filters, persist)


def yolo_image_inference(image: Image.Image, conf_level: float, image_size: int, filters: str = None, persist: bool = True):

    filter_list = []
    if filters:
        filter_list = [f.strip() for f in filters.split(",") if f.strip()]


    names = getattr(model, 'names', None)
    if names is None or (isinstance(names, (list, tuple)) and not names):
        # If model doesn't expose names, fall back to provided classes list
        names = classes_list if classes_list else {}

    try:
        try:
            # Run prediction. Note: ultralytics returns a list of Results
            results = model.predict(image, conf=conf_level, device=device, imgsz=image_size, verbose=False)
            results = apply_filter(results, filter_list, names)

            if persist == True:

                image = annotate_image_pil(image, results, names)

                id = str(uuid.uuid4())
                image.save(f"./tmp/{id}.png")

        except Exception as e:
            logging.warn( f"Inference failed for: {e}")

    except Exception as e:
        logging.error( f"decoding image: {e}")
        return
    return build_detection_response(image,results,names)


def apply_filter(results, filter_list, names):
    """
    Removes detections whose class name is inside filter_list.

    results: list returned by a YOLO model
    filter_list: list of class names to exclude, e.g. ["person", "cup"]
    names: model.names (mapping class_id -> class_name)
    """

    filtered_results = []

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        keep_mask = []
        for cls_id in r.boxes.cls.cpu().numpy().astype(int):
            class_name = names[cls_id]
            keep_mask.append(class_name not in filter_list)

        # Convert boolean mask to indices
        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]

        # Create a shallow copy of the result with filtered boxes
        new_r = r
        new_r.boxes = r.boxes[keep_indices]  # ultralytics supports indexing this way
        filtered_results.append(new_r)

    return filtered_results


def build_detection_response(
    img: Image.Image,
    results,
    names
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dict with detection results.

    Structure:
    {
        "image": {
            "width": int,
            "height": int
        },
        "detections": [
            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": { "x1": int, "y1": int, "x2": int, "y2": int },
                "color": [int, int, int]  # same color as annotate_image_pil
            },
            ...
        ]
    }
    """
    width, height = img.size
    detections: List[Dict[str, Any]] = []

    for r in results:
        if not hasattr(r, 'boxes') or r.boxes is None:
            continue

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)

        for (box, conf, cls_id) in zip(xyxy, confs, clss):
            cls_id = int(cls_id)

            # Resolve class name
            if isinstance(names, dict):
                name = names.get(cls_id, str(cls_id))
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                name = str(names[cls_id])
            else:
                name = str(cls_id)

            # Ensure JSON-serializable types
            x1, y1, x2, y2 = [int(v) for v in box]
            confidence = float(conf)
            color = get_color_for_class(cls_id)  # (R, G, B)

            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": confidence,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "color": list(color),
            })

    response: Dict[str, Any] = {
        "image": {
            "width": width,
            "height": height,
        },
        "detections": detections,
    }
    return response





def annotate_image_pil(img: Image.Image, results, names) -> Image.Image:
    """
    Draw boxes with labels and confidences on a Pillow image.
    `names` can be a dict (id->name) or a list/tuple (index->name).
    Ultralytics results: boxes.xyxy, boxes.conf, boxes.cls
    """
    # Ensure we can draw in RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # Pillow will fall back internally

    for r in results:
        if not hasattr(r, 'boxes') or r.boxes is None:
            continue
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)

        for (box, conf, cls_id) in zip(xyxy, confs, clss):
            cls_id = int(cls_id)
            if isinstance(names, dict):
                name = names.get(cls_id, str(cls_id))
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                name = str(names[cls_id])
            else:
                name = str(cls_id)

            label = f"{name} {conf:.2f}"
            color = get_color_for_class(cls_id)  # (R, G, B)
            draw_box_with_label_pil(img, draw, box, label, color, font)

    return img


def get_color_for_class(cls_id: int) -> tuple:
    """Deterministic pseudo-color for a class id (R, G, B)."""
    np.random.seed(cls_id + 12345)
    color = np.random.randint(60, 256, size=3).tolist()  # avoid too-dark colors
    return int(color[0]), int(color[1]), int(color[2])


def draw_box_with_label_pil(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box,
    label: str,
    color=(0, 255, 0),
    font=None,
):
    x1, y1, x2, y2 = map(int, box)

    # Bounding box
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

    # Measure text
    if font is None:
        font = ImageFont.load_default()

    # Newer Pillow: textbbox, older: textsize
    try:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
    except AttributeError:
        tw, th = draw.textsize(label, font=font)

    pad = 3
    th_box = th + 2 * pad

    y1_text = max(0, y1 - 4)
    top_box = y1_text - th_box if y1_text - th_box > 0 else y1

    # Background rectangle for text
    box_bg = [
        (x1, max(0, top_box)),
        (x1 + tw + 2 * pad, max(0, top_box) + th_box),
    ]
    draw.rectangle(box_bg, fill=color)

    # Text (black on colored bg)
    text_pos = (x1 + pad, max(0, top_box) + pad)
    draw.text(text_pos, label, fill=(0, 0, 0), font=font)
