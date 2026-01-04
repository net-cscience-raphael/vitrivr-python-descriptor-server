import base64
import json
from io import BytesIO

import open_clip
import torch
from PIL import Image
from apiflask import APIBlueprint
from flask import request
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

@open_clip_influence_bp.post("/extract/clip_influence")
@open_clip_influence_bp.doc(
    summary="")
def clip_influence_image_text():
    data = request.form.get('data-img', '')
    _, img_enc = data.split("base64,", 1)
    data = request.form.get('data-txt', '')
    _, txt_enc = data.split("utf-8,", 1)


    try:
        text = base64.b64decode(txt_enc).decode("utf-8")
        image = Image.open(BytesIO(base64.b64decode(img_enc)))

        response = cmi.imageTextVectorPair_response(image, text)

    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    return response


@open_clip_influence_bp.post("/extract/clip_image_from_cache")
@open_clip_influence_bp.doc(
    summary="CLIP endpoint for feature extraction on image, where the image is transmitted in the body by a data URL")
def clip_image_from_cache():
    image_id = str(request.form.get('image_id'))


    try:
        image =  image_cache.get(image_id)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    feature = "[]" if image is None else json.dumps(feature_image(image).tolist())
    return feature



def feature_image(image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        gc.collect()
        return image_features.cpu().numpy().flatten()
