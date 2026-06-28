from ultralytics import YOLO
import cv2
import numpy as np


class BehaviourDetector:
    def __init__(self, model_size="yolov8n"):
        """
        model_size options:
        - yolov8n: fastest
        - yolov8s: balanced
        - yolov8m: more accurate
        """

        print(f"Loading {model_size} model...")

        self.model = YOLO(f"{model_size}.pt")

        self.relevant_classes = {
            0: "person",
            67: "cell phone",
        }

    def detect_frame(self, frame, confidence_threshold=0.5):

        results = self.model(
            frame,
            conf=confidence_threshold,
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                cls_id = int(box.cls[0])

                if cls_id in self.relevant_classes:

                    detections.append({
                        "class": self.relevant_classes[cls_id],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    })

        return detections

    def draw_detections(self, frame, detections):

        for det in detections:

            x1, y1, x2, y2 = [
                int(v)
                for v in det["bbox"]
            ]

            colour = (
                (0, 255, 0)
                if det["class"] == "person"
                else (0, 0, 255)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colour,
                2
            )

            label = (
                f'{det["class"]} '
                f'{det["confidence"]:.2f}'
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                2
            )

        return frame


if __name__ == "__main__":

    detector = BehaviourDetector("yolov8n")

    test_image = cv2.imread(
        "data/test/sample_frame.jpg"
    )

    if test_image is None:

        print(
            "ERROR: Could not load test image."
        )

    else:

        detections = detector.detect_frame(
            test_image
        )

        print(
            f"Detected {len(detections)} objects:"
        )

        print(detections)

        output = detector.draw_detections(
            test_image.copy(),
            detections
        )

        cv2.imwrite(
            "data/test/result.jpg",
            output
        )

        print(
            "Result saved to "
            "data/test/result.jpg"
        )