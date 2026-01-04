from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .MaskingGenerator import MaskingGenerator as MaskingGenerator

class FilterProvider(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def filter_fnc(self, generator: "MaskingGenerator"):
        pass
