import cv2

from src.pose.pose_estimator import (
    PoseEstimator
)

pose = PoseEstimator()

img = cv2.imread(
    "data/test/sample_frame.jpg"
)

if img is None:

    print(
        "Image not found."
    )

else:

    keypoints = pose.extract_keypoints(
        img
    )

    if keypoints is None:

        print(
            "No person detected."
        )

    else:

        print(
            "Keypoint shape:",
            keypoints.shape
        )

        print(
            "Head orientation:",
            pose.get_head_orientation(
                keypoints
            )
            
        )
        print(keypoints)
        keypoints = pose.extract_keypoints(img)