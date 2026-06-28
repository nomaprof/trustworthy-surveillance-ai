class BehaviourClassifier:
    """
    Rule-based behaviour classifier using extracted features.
    """

    BEHAVIOURS = [
        "NORMAL",
        "PHONE_USE",
        "HEAD_TURNING",
        "SUSPICIOUS_POSTURE"
    ]

    def classify(self, features):
        """
        Classify behaviour using extracted features.

        Parameters
        ----------
        features : dict
            Dictionary produced by FeatureExtractor.

        Returns
        -------
        behaviour : str
        confidence : float
        explanation : dict
        """

        behaviour = "NORMAL"
        confidence = 0.90
        explanation = {}

        # -----------------------------
        # Rule 1: Phone detected
        # -----------------------------
        if (
            features["phone_detected"] == 1 and
            features["phone_confidence"] > 0.60
        ):

            behaviour = "PHONE_USE"
            confidence = features["phone_confidence"]

            explanation["reason"] = "Cell phone detected"

            explanation["phone_confidence"] = (
                features["phone_confidence"]
            )

        # -----------------------------
        # Rule 2: Head turned
        # -----------------------------
        elif (
            features["head_asymmetry"] > 0.35
        ):

            behaviour = "HEAD_TURNING"

            confidence = min(
                0.5 + features["head_asymmetry"],
                0.95
            )

            explanation["reason"] = (
                "Head turned significantly"
            )

            explanation["head_asymmetry"] = (
                features["head_asymmetry"]
            )

        return (
            behaviour,
            confidence,
            explanation
        )