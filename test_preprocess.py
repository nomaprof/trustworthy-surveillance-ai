import cv2
from src.utils.preprocess import preprocess_frame

img = cv2.imread(
    "data/splits/cheating/train/Normal/istockphoto-462641909-612x612 - Copy.jpg"
)

processed = preprocess_frame(img)

print(processed.shape)
print(processed.min())
print(processed.max())