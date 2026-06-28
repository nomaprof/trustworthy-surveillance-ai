import cv2

from src.utils.augmentation import (
    add_gaussian_noise,
    reduce_brightness,
    add_occlusion
)

img = cv2.imread(
    "data/splits/cheating/train/Normal/istockphoto-462641909-612x612 - Copy.jpg"
)

noise = add_gaussian_noise(img.copy())
dark = reduce_brightness(img.copy())
occ = add_occlusion(img.copy())

cv2.imwrite("noise.jpg", noise)
cv2.imwrite("dark.jpg", dark)
cv2.imwrite("occlusion.jpg", occ)

print("Saved augmentation examples.")