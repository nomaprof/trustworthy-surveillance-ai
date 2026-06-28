import cv2
import numpy as np

def preprocess_frame(frame, target_size=(640, 640)):

    h, w = frame.shape[:2]

    scale = min(
        target_size[0] / h,
        target_size[1] / w
    )

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2

    padded = cv2.copyMakeBorder(
        resized,
        top,
        target_size[0] - new_h - top,
        left,
        target_size[1] - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    normalised = padded.astype(np.float32) / 255.0

    return normalised