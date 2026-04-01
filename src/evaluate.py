"""
Stage 5 — Model Evaluation
Evaluate best model on held-out test set and save metrics + plots.
"""
import os, pickle, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, f1_score
)


def main():
    proc      = os.path.join("data", "processed")
    model_dir = "models"
    reports   = "reports"
    os.makedirs(reports, exist_ok=True)

    with open(os.path.join(model_dir, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(os.path.join(proc, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(proc, "y_test.csv")).squeeze()

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_t = thresholds[np.argmax([f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds])]

    test_metrics = {
        "roc_auc":          round(roc_auc, 4),
        "best_threshold":   round(float(best_t), 2),
        "f1_at_threshold":  round(f1_score(y_test, (y_proba >= best_t).astype(int)), 4),
    }
    with open(os.path.join(reports, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── ROC CSV (for DVC plots) ───────────────────────────────────────────────
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(reports, "roc_curve.csv"), index=False)

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix — {type(model).__name__}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(reports, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Evaluation complete")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc:.4f} | Best threshold: {best_t:.2f}")


if __name__ == "__main__":
    main()
