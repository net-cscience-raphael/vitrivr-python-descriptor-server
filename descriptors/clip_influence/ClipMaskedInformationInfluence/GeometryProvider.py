from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import torch

if TYPE_CHECKING:
    from .MaskingGenerator import MaskingGenerator


class GeometryProvider(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def geometry_fnc(self, generator: "MaskingGenerator", batch_img_tensor: torch.Tensor) -> torch.Tensor:
        pass
