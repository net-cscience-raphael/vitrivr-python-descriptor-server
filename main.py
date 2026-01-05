import argparse
import logging

from apiflask import APIFlask
from flask import jsonify

from util.CustomLogger import setup_logging

app = APIFlask(__name__, title='ExternalPython Descriptor Server API for vitrivr', version='1.0.0')


MEGABYTE = (2 ** 10) ** 2
app.config['MAX_CONTENT_LENGTH'] = None
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * MEGABYTE

@app.route("/health", methods=["GET"])  # Add thisAdd commentMore actions
def health():
    return jsonify(status="ok")

# import necessary modules
from descriptors.open_clip_lion_text import open_clip_lion_text
from descriptors.open_clip_lion_image import open_clip_lion_image
from descriptors.ocr import ocr
from descriptors.yolo_world_image import yolo_world_image
from descriptors.image_cache_endpoint import image_cache_bp
from descriptors.yolo_world_image_semantic import yolo_world_image_semantic_bp
from descriptors.clip_influence.open_clip_influence import open_clip_influence_bp


# specify here all modules, that will be needed for feature extraction server

def register_modules():
    app.register_blueprint(open_clip_lion_text)
    app.register_blueprint(open_clip_lion_image)
    #app.register_blueprint(dino_v2)
    app.register_blueprint(ocr)
    #app.register_blueprint(asr_whisper)
    #app.register_blueprint(emotions)
    app.register_blueprint(yolo_world_image)
    app.register_blueprint(yolo_world_image_semantic_bp)
    app.register_blueprint(image_cache_bp)
    app.register_blueprint(open_clip_influence_bp)


def entrypoint(host, port, args):
    app.run(host=host, port=port)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

loudness = parser.add_mutually_exclusive_group()
loudness.add_argument("-v", "--verbose", action="store_true")
loudness.add_argument("-q", "--quiet", action="store_true")

parser.add_argument('--host', type=str, help='Host to connect to.', default='localhost')
parser.add_argument('--port', type=int, help='Port to listen on.', default=8888)
parser.add_argument('--device', type=str, help='Device to use for feature extraction.', default='gpu')

args = parser.parse_args()
device = args.device

if __name__ == '__main__':
    register_modules()
    setup_logging(args)
    logging.info("Starting server...")
    entrypoint(host=args.host, port=args.port, args=args.device)
