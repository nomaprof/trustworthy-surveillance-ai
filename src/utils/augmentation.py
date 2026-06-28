import cv2, numpy as np

def add_gaussian_noise(frame, std=25):
    noise = np.random.normal(0, std, frame.shape).astype(np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def reduce_brightness(frame, factor=0.4):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def add_occlusion(frame, occlusion_ratio=0.3):
    h, w = frame.shape[:2]
    occ_h = int(h * occlusion_ratio)
    occ_w = int(w * occlusion_ratio)
    y = np.random.randint(0, h - occ_h)
    x = np.random.randint(0, w - occ_w)
    frame[y:y+occ_h, x:x+occ_w] = 128  # Grey block
    return frame
