import numpy as np
from skimage import img_as_float32, img_as_ubyte, img_as_uint


def as_dtype(img, dtype) -> np.ndarray:
    """Convert an image to a specific fundamental dtype"""
    if dtype == np.uint8:
        return img_as_ubyte(img)
    elif dtype == np.uint16:
        return img_as_uint(img)
    elif dtype == np.float32:
        return img_as_float32(img)
