class ExplanationEngine:
    """
    Generates human-readable explanations for
    behaviour classification decisions.
    """

    def generate(
        self,
        behaviour,
        confidence,
        features,
        explanation
    ):

        report = {}

        report["Behaviour"] = behaviour

        report["Confidence"] = round(
            confidence,
            2
        )

        report["Reasons"] = []

        if features["phone_detected"]:

            report["Reasons"].append(
                f"Cell phone detected "
                f"(confidence = "
                f"{features['phone_confidence']:.2f})"
            )

        if features["head_asymmetry"] > 0.35:

            report["Reasons"].append(
                f"Head asymmetry = "
                f"{features['head_asymmetry']:.2f}"
            )

        if not report["Reasons"]:

            report["Reasons"].append(
                "No suspicious behaviour detected."
            )

        return report