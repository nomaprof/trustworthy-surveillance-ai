from src.features.feature_extractor import FeatureExtractor
from src.detection.classifier import BehaviourClassifier
from src.explainability.explanation_engine import ExplanationEngine
from src.accountability.logger import AccountabilityLogger




# Simulated detections from YOLO
detections = [
    {
        "class": "person",
        "confidence": 0.98,
        "bbox": [100, 100, 400, 600]
    },
    {
        "class": "cell phone",
        "confidence": 0.82,
        "bbox": [220, 320, 260, 380]
    }
]

# Simulated pose output
head_asymmetry = 0.15

# Create objects
feature_extractor = FeatureExtractor()
classifier = BehaviourClassifier()
explainer = ExplanationEngine()
logger = AccountabilityLogger()

# Extract features
features = feature_extractor.extract_features(
    detections,
    head_asymmetry
)

print("Extracted Features:")
print(features)

# Classify behaviour
behaviour, confidence, explanation = classifier.classify(
    features
)
report = explainer.generate(
    behaviour,
    confidence,
    features,
    explanation
)

logger.log(
    behaviour,
    confidence,
    features,
    report
)

print("\nClassification Result")
print("---------------------")

for key, value in report.items():
    print(f"{key}: {value}")