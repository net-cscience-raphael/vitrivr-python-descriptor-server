from typing import TYPE_CHECKING

import torch

from .GeometryProvider import GeometryProvider

if TYPE_CHECKING:
    from .MaskingGenerator import MaskingGenerator


class SquareProvider(GeometryProvider):
    def __init__(self, **kwargs):
        self.geometry_size = kwargs["geometry_size"]  # (h_rel, w_rel)

    def geometry_fnc(self, generator: "MaskingGenerator", batch_img_tensor: torch.Tensor) -> torch.Tensor:
        # batch_img_tensor is [C,H,W] in your usage (a single image)
        cx, cy = generator.get_xy_pixel_point()

        # fixed pixel window sizes
        win_w = max(1, round(self.geometry_size[1] * generator.image_w))
        win_h = max(1, round(self.geometry_size[0] * generator.image_h))

        # top-left from center (use half sizes)
        x0 = cx - win_w // 2
        y0 = cy - win_h // 2

        # clamp so window stays inside image while keeping size (when possible)
        x0 = max(0, min(generator.image_w - win_w, x0))
        y0 = max(0, min(generator.image_h - win_h, y0))

        x1 = x0 + win_w
        y1 = y0 + win_h

        return batch_img_tensor[:, y0:y1, x0:x1]
