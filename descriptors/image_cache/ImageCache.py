# descriptors/image_cache.py
import threading
import uuid
from collections import OrderedDict
from typing import Optional

from PIL import Image


class ImageCache:
    """
    Simple in-memory LRU image cache.
    Maps UUID (or custom string) -> PIL.Image.Image
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, max_items: int = 1024):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init(max_items)
            return cls._instance

    def _init(self, max_items: int):
        self.max_items = max_items
        self._cache: dict[str, Image.Image] = {}
        self._lru = OrderedDict()
        self._lock = threading.Lock()

    def put(self, img: Image.Image, key: Optional[str] = None) -> str:
        """
        Store an image in the cache and return its key.

        If `key` is None, a new UUID is generated.
        If `key` already exists, the image is replaced and the entry becomes MRU.
        """
        if key is None:
            key = str(uuid.uuid4())

        with self._lock:
            # If new key and cache is full -> evict LRU
            if key not in self._cache and len(self._cache) >= self.max_items:
                old_key, _ = self._lru.popitem(last=False)
                self._cache.pop(old_key, None)

            self._cache[key] = img
            self._lru[key] = None
            self._lru.move_to_end(key, last=True)  # mark as most recently used

        return key

    def get(self, key: str) -> Optional[Image.Image]:
        with self._lock:
            img = self._cache.get(key)
            if img is not None:
                self._lru.move_to_end(key, last=True)
            return img

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._cache
            self._cache.pop(key, None)
            self._lru.pop(key, None)
            return existed

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._lru.clear()
