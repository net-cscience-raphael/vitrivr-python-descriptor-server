from typing import TYPE_CHECKING

import torch

from .FilterProvider import FilterProvider

if TYPE_CHECKING:
    from .MaskingGenerator import MaskingGenerator as MaskingGenerator

class MeanFilterProvider(FilterProvider):


    def __init__(self, **kwargs):
        self.variance = kwargs.get("variance", 0.5)
        pass


    def filter_fnc(self, generator: "MaskingGenerator") -> torch.Tensor:
        window :torch.Tensor = generator.geometry_fnc(generator.base_img_tensor[0])
        return window + (self.variance**0.5) * torch.randn(window.size()).to(generator.device)
