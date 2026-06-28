import mediapipe as mp
import cv2
import numpy as np


class PoseEstimator:

    def __init__(self):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

    def extract_keypoints(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(
            rgb_frame
        )

        if results.pose_landmarks is None:
            return None

        keypoints = []

        for lm in results.pose_landmarks.landmark:

            keypoints.append([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])

        return np.array(keypoints)

    def get_head_orientation(
        self,
        keypoints
    ):

        if keypoints is None:
            return None

        nose = keypoints[0, :2]
        left_ear = keypoints[7, :2]
        right_ear = keypoints[8, :2]

        dist_left = np.linalg.norm(
            nose - left_ear
        )

        dist_right = np.linalg.norm(
            nose - right_ear
        )

        asymmetry = abs(
            dist_left - dist_right
        ) / (
            dist_left +
            dist_right +
            1e-6
        )

        return asymmetry