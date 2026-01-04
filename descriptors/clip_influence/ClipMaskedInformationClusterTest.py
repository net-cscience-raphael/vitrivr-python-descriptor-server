import argparse

import numpy as np
import torch
from PIL import Image

from descriptors.clip_influence.ClipMaskedInformationInfluence import MaskingMode, ClipMaskedInformationCluster, \
    tensor_to_pil, show_overlay_any


def main(args):
    # Load and preprocess image
    kwargs = {
        "geometry_size": (1 / 3, 1 / 3),  # relative to image size
        "step_size": (1 / 3, 1 / 3),  # relative to image size
        "start_point": (1 / 6, 1 / 6),  # pixel offset relative to image size
        "steps": (3, 3),  # number of steps in h,w
        "mode": MaskingMode.KEEP_ONLY
    }
    cmi = ClipMaskedInformationCluster(**kwargs)

    iterations = zip(args.image_paths, args.labels)
    for image_path, label in iterations:
        img = Image.open(image_path).convert("RGB")
        scores, point_hard, point_soft, generator = cmi.imageTextPair(img, label)
        deltas_np = scores.cpu().numpy()
        max_index = np.argmax(deltas_np)
        # batch_img_f_masked[max_index] = 0.0  # zero-out most influential region
        # deltas = influence_calculator(img_f_base, batch_img_f_masked, txt_f,mode)
        # deltas = deltas
        img_out = tensor_to_pil(generator.batch_img_tensor[max_index], cmi.clip_service.preprocessor)
        show_overlay_any(img_out, scores, generator, alpha=0.5, label=label)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--image-paths", type=str, nargs="+",
                   default=[
                       "../../Resources/cornflakes.webp",
                       "../../Resources/cornflakes.webp",
                       "../../Resources/kitchen2.webp",
                       "../../Resources/kitchen2.webp",
                       "../../Resources/sample1.jpg",
                       "../../Resources/sample1.jpg",
                       "../../Resources/sample1.jpg",
                       "../../Resources/astronaut.png",
                       "../../Resources/astronaut.png",
                       "../../Resources/catdog.png",
                       "../../Resources/catdog.png",
                       "../../Resources/dogbird.png",
                       "../../Resources/dogbird.png",
                       "../../Resources/shop1.webp",
                       "../../Resources/shop1.webp",
                       "../../Resources/shop1.webp",
                       "../../Resources/shop1.webp",
                       "../../Resources/shop1.webp",
                   ])

    p.add_argument("--labels", type=str, nargs="+",
                   default=[
                       "cornflakes on a table",
                       "a tea kettle on a cooking stove",
                       "a fridge with drawings",
                       "a tea kettle on a cooking stove",
                       "photo of a dog",
                       "a golden dog",
                       "a cat on the sofa",
                       "an astronaut with orange suit",
                       "a spaceshuttle model in background",
                       "a cat and a dog",
                       "a dog",
                       "a bird",
                       "a dog",
                       "a wheelchair",
                       "a plant beside a golden vase",
                       "photos on a wall",
                       "ceramics on the floor",
                       "a red carped on the floor",
                   ])

    main(p.parse_args())
