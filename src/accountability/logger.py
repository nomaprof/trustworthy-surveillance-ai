import json
import os
from datetime import datetime


class AccountabilityLogger:
    """
    Records every system decision
    in a JSON audit log.
    """

    def __init__(self, log_file="results/audit_log.json"):

        self.log_file = log_file

        os.makedirs(
            os.path.dirname(log_file),
            exist_ok=True
        )

        if not os.path.exists(log_file):

            with open(log_file, "w") as f:
                json.dump([], f)

    def log(
        self,
        behaviour,
        confidence,
        features,
        explanation
    ):

        record = {

            "timestamp": datetime.now().isoformat(),

            "behaviour": behaviour,

            "confidence": round(
                confidence,
                3
            ),

            "features": features,

            "explanation": explanation
        }

        with open(self.log_file, "r") as f:

            logs = json.load(f)

        logs.append(record)

        with open(self.log_file, "w") as f:

            json.dump(
                logs,
                f,
                indent=4
            )

        print(
            "Decision logged successfully."
        )