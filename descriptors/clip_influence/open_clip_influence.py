import base64
import json, ast
from io import BytesIO

import numpy as np
import open_clip
import torch
from PIL import Image
from apiflask import APIBlueprint
from flask import request, jsonify
import gc

from descriptors.clip_influence.ClipMaskedInformationInfluence.ClipMaskedInformationCluster import \
    ClipMaskedInformationCluster
from descriptors.clip_influence.ClipMaskedInformationInfluence.MaskingGenerator import MaskingMode
from descriptors.clip_influence.ClipMaskedInformationInfluence.OpenClipScoringService import OpenClipScoringService
from descriptors.image_cache.ImageCache import ImageCache

#from main import device #FIXME cyclic dependency
open_clip_influence_bp = APIBlueprint('open_clip_influence_bp', __name__)
image_cache = ImageCache()

kwargs = {
    "geometry_size": (1 / 3, 1 / 3),  # relative to image size
    "step_size": (1 / 3, 1 / 3),  # relative to image size
    "start_point": (1 / 6, 1 / 6),  # pixel offset relative to image size
    "steps": (3, 3),  # number of steps in h,w
    "mode": MaskingMode.KEEP_ONLY
}
cmi = ClipMaskedInformationCluster(**kwargs)

@open_clip_influence_bp.post("/extract/clip_influence_image_text_scores")
@open_clip_influence_bp.doc(
    summary="")
def clip_influence_image_text_scores():
    data = request.form.get('data-img', '')
    _, img_enc = data.split("base64,", 1)
    data = request.form.get('data-txt', '')
    _, txt_enc = data.split("utf-8,", 1)


    try:
        text = base64.b64decode(txt_enc).decode("utf-8")
        image = Image.open(BytesIO(base64.b64decode(img_enc)))

        response = cmi.imageTextPair_response(image, text)

    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    return response

@open_clip_influence_bp.post("/extract/clip_influence_image_embedded")
@open_clip_influence_bp.doc(
    summary="")
def clip_influence_image_embedded():
    data = request.form.get('data-img', '')
    _, img_enc = data.split("base64,", 1)

    try:

        image = Image.open(BytesIO(base64.b64decode(img_enc)))

        response = cmi.imageVectors_response(image)

    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    return response


@open_clip_influence_bp.post("/extract/clip_influence_img_f_text_scores")
@open_clip_influence_bp.doc(
    summary="")
def clip_influence_img_f_text_scores():


    data = request.form.get('data-txt', '')
    _, txt_enc = data.split("utf-8,", 1)

    vec_field = request.form.get("data-vec-img", "")
    prefix, payload  = vec_field.split(",", 1)
    if prefix != "vector/float32;base64":
        return jsonify(error="unsupported vector encoding"), 400

    vec_field = request.form.get("data-vec-base", "")
    prefix, payload2  = vec_field.split(",", 1)

    if prefix != "vector/float32;base64":
        return jsonify(error="unsupported vector encoding"), 400

    raw = base64.b64decode(payload)
    text = raw.decode("utf-8")
    img_f = np.array(json.loads(''.join(text.split()))).astype(np.float32)

    raw = base64.b64decode(payload2)
    text = raw.decode("utf-8")
    img_f_base = np.array(json.loads(''.join(text.split()))).astype(np.float32)

    processed_image_w = request.form.get('image_w', 224)
    processed_image_h = request.form.get('image_h', 224)
    try:
        img_f = torch.from_numpy(img_f).to(cmi.clip_service.device)
        txt_f = cmi.embedd_text(text=base64.b64decode(txt_enc).decode("utf-8"))
        response = cmi.imageTextVectorPair_response(img_f_base, img_f, txt_f, processed_image_w, processed_image_h)

    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    return response