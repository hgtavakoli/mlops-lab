"""Image preprocessing shared by calibration and inference (same as in step 3)."""
import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping the aspect ratio, then pad to size x size with grey (114)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 0, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


def to_tensor(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR image -> float32 NCHW tensor in [0, 1], RGB order, batch size 1."""
    img = letterbox(img_bgr, size)[:, :, ::-1]  # BGR -> RGB
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
