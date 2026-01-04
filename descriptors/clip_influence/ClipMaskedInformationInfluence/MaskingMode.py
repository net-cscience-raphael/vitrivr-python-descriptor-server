from enum import Enum


class MaskingMode(str, Enum):
    MASK_OUT = "mask_out"     # mask the region inside an otherwise intact image
    KEEP_ONLY = "keep_only"   # keep the region; mask everything else