import numpy as np


class FeatureExtractor:
    """
    Converts detections and pose information into
    numerical features for behaviour classification.
    """

    def extract_features(self, detections, head_asymmetry):
        phone_detected = 0
        phone_confidence = 0.0
        person_detected = 0

        for det in detections:

            if det["class"] == "person":
                person_detected = 1

            elif det["class"] == "cell phone":
                phone_detected = 1
                phone_confidence = max(
                    phone_confidence,
                    det["confidence"]
                )

        if head_asymmetry is None:
            head_asymmetry = 0.0

        features = {
            "person_detected": person_detected,
            "phone_detected": phone_detected,
            "phone_confidence": float(phone_confidence),
            "head_asymmetry": float(head_asymmetry)
        }

        return features