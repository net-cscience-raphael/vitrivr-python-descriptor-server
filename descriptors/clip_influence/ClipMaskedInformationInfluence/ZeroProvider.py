from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from .FilterProvider import FilterProvider

if TYPE_CHECKING:
    from .MaskingGenerator import MaskingGenerator as MaskingGenerator

class ZeroProvider(FilterProvider):


    def __init__(self, **kwargs):
        pass


    def filter_fnc(self, generator: "MaskingGenerator") -> float:
        return 0.0

