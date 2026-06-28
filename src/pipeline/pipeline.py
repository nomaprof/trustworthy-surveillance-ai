import cv2

from src.detection.detector import BehaviourDetector
from src.pose.pose_estimator import PoseEstimator
from src.features.feature_extractor import FeatureExtractor
from src.detection.classifier import BehaviourClassifier
from src.explainability.explanation_engine import ExplanationEngine
from src.accountability.logger import AccountabilityLogger

class TrustworthySurveillancePipeline:

    def __init__(self):

        print("Initialising pipeline...")

        self.detector = BehaviourDetector("yolov8n")

        self.pose_estimator = PoseEstimator()

        self.feature_extractor = FeatureExtractor()

        self.classifier = BehaviourClassifier()

        self.explainer = ExplanationEngine()

        self.logger = AccountabilityLogger()

        print("Pipeline ready.")

if __name__ == "__main__":

    pipeline = TrustworthySurveillancePipeline()
        