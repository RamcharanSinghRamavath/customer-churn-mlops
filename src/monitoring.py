"""
Model Monitoring Module
========================
Detects data drift between reference (training) and production data.
Uses statistical tests — no Evidently required.
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def kolmogorov_smirnov_drift(ref: pd.Series, cur: pd.Series, alpha: float = 0.05) -> dict:
    """KS test for numerical drift detection."""
    stat, p_value = stats.ks_2samp(ref.dropna(), cur.dropna())
    return {"statistic": round(stat, 4), "p_value": round(p_value, 4), "drift": p_value < alpha}


def chi_square_drift(ref: pd.Series, cur: pd.Series, alpha: float = 0.05) -> dict:
    """Chi-square test for categorical drift detection."""
    cats = set(ref.unique()) | set(cur.unique())
    ref_counts = ref.value_counts().reindex(cats, fill_value=0)
    cur_counts = cur.value_counts().reindex(cats, fill_value=0)
    stat, p_value = stats.chisquare(f_obs=cur_counts, f_exp=ref_counts * len(cur) / len(ref))
    return {"statistic": round(float(stat), 4), "p_value": round(float(p_value), 4), "drift": p_value < alpha}


def run_drift_report(production_data: pd.DataFrame = None) -> dict:
    """
    Compare production data to training reference.
    If no production data provided, simulates drift by perturbing training data.
    """
    reference = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))

    if production_data is None:
        # Simulate slight drift for demo
        production_data = reference.copy()
        production_data["MonthlyCharges"] += np.random.normal(5, 2, len(production_data))
        production_data["tenure"] = (production_data["tenure"] * 0.85).astype(int)
        logger.info("ℹ️  Using simulated production data (with artificial drift)")

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    report = {"timestamp": datetime.now().isoformat(), "drift_detected": False, "features": {}}

    for col in num_cols:
        if col in reference.columns and col in production_data.columns:
            result = kolmogorov_smirnov_drift(reference[col], production_data[col])
            report["features"][col] = result
            if result["drift"]:
                report["drift_detected"] = True
                logger.warning(f"⚠️  Drift detected in '{col}' (p={result['p_value']})")
            else:
                logger.info(f"✅ No drift in '{col}' (p={result['p_value']})")

    # Save report
    report_path = os.path.join(DATA_DIR, "drift_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"📄 Drift report saved: {report_path}")
    return report


if __name__ == "__main__":
    report = run_drift_report()
    print(json.dumps(report, indent=2))
