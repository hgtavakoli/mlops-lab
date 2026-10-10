import numpy as np

from lablib.preprocess import letterbox, to_tensor


def test_letterbox_keeps_aspect_and_pads_grey():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)  # wide white image
    out = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert (out[0, 0] == 114).all()          # top padding is grey
    assert (out[160, 160] == 255).all()      # centre is the image


def test_to_tensor_shape_range_and_rgb_order():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 2] = 255                       # pure red in OpenCV's BGR order
    x = to_tensor(img, 320)
    assert x.shape == (1, 3, 320, 320)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert x[0, 0, 160, 160] == 1.0          # channel 0 is red after BGR -> RGB
