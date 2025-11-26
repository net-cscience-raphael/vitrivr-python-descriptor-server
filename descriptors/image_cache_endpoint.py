# descriptors/image_cache_endpoint.py
import base64
from io import BytesIO

from apiflask import APIBlueprint
from flask import request, jsonify
from PIL import Image

from descriptors.image_cache.ImageCache import ImageCache

image_cache_bp = APIBlueprint('image_cache', __name__)
cache = ImageCache(max_items=2048)


@image_cache_bp.post("/cache/image")
@image_cache_bp.doc(
    summary="Store an image in memory and get back an ID",
    description=(
        "Send an image as data URL via form field 'data'.\n"
        "Optional: form field 'id' to force a specific key (e.g. your own UUID).\n"
        "Returns JSON { 'id': '<id-used>' }."
    )
)
def cache_image():
    """
    Expects:
      - form field 'data' = data URL (e.g. 'data:image/png;base64,...')
      - optional form field 'id' = custom key to use in cache
    """
    data = request.form.get('data', '')
    custom_id = request.form.get('id')  # optional

    if "base64," not in data:
        return jsonify(error="Missing or invalid data URL"), 400

    try:
        _, encoded = data.split("base64,", 1)
        img = Image.open(BytesIO(base64.b64decode(encoded)))
    except Exception as e:
        print(f"[ERROR] Failed to decode image for cache: {e}")
        return jsonify(error="Failed to decode image"), 400

    used_id = cache.put(img, key=custom_id)
    return jsonify(id=used_id)
