import json
import os


def check_alerts():
    print("Evaluating Alerts and Retrain Triggers...")

    reports_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    drift_path = os.path.join(reports_dir, "drift_report.json")
    perf_path = os.path.join(reports_dir, "performance_report.json")

    retrain_required = False
    reasons = []

    if os.path.exists(drift_path):
        with open(drift_path, "r") as f:
            drift = json.load(f)

        for feature, metrics in drift.get("features", {}).items():
            if metrics.get("drift_detected", False):
                retrain_required = True
                reasons.append(f"Drift detected in {feature} (PSI = {metrics.get('psi')})")

    if os.path.exists(perf_path):
        with open(perf_path, "r") as f:
            perf = json.load(f)

        if perf.get("performance_degraded", False):
            retrain_required = True
            reasons.append(f"Model performance degraded (MAE = {perf.get('price_mae')})")

    if retrain_required:
        print("ALERT: Retrain Triggered!")
        for r in reasons:
            print(f" - {r}")

        out_path = os.path.join(reports_dir, "retrain_required.json")
        with open(out_path, "w") as f:
            json.dump({"retrain": True, "reasons": reasons}, f, indent=2)
    else:
        print("System Healthy. No retrain required.")


if __name__ == "__main__":
    check_alerts()
