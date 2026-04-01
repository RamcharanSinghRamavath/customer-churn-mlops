"""
Stage 4 — Model Training
Train 3 models, track with MLflow, save best model.
"""
import os, pickle, json
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
}


def get_metrics(model, X, y):
    p  = model.predict(X)
    pr = model.predict_proba(X)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y, p), 4),
        "precision": round(precision_score(y, p), 4),
        "recall":    round(recall_score(y, p), 4),
        "f1":        round(f1_score(y, p), 4),
        "roc_auc":   round(roc_auc_score(y, pr), 4),
    }


def main():
    proc      = os.path.join("data", "processed")
    model_dir = "models"
    reports   = "reports"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports,   exist_ok=True)

    X_train = pd.read_csv(os.path.join(proc, "X_train_smote.csv"))
    y_train = pd.read_csv(os.path.join(proc, "y_train_smote.csv")).squeeze()
    X_val   = pd.read_csv(os.path.join(proc, "X_val_scaled.csv"))
    y_val   = pd.read_csv(os.path.join(proc, "y_val.csv")).squeeze()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Customer_Churn_Prediction")

    best_model, best_f1, best_name = None, 0, ""
    all_metrics = {}

    for name, model in MODELS.items():
        print(f"\n🔄 Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            val_m = get_metrics(model, X_val, y_val)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
            mlflow.sklearn.log_model(model, "model", registered_model_name=f"Churn_{name}")
            all_metrics[name] = val_m
            print(f"   F1={val_m['f1']} | AUC={val_m['roc_auc']}")
            if val_m["f1"] > best_f1:
                best_f1, best_model, best_name = val_m["f1"], model, name

    with open(os.path.join(model_dir, "best_model.pkl"),      "wb") as f: pickle.dump(best_model, f)
    with open(os.path.join(model_dir, "feature_columns.pkl"), "wb") as f: pickle.dump(list(X_train.columns), f)
    with open(os.path.join(reports, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, **all_metrics[best_name]}, f, indent=2)

    print(f"\n🏆 Best: {best_name} | F1={best_f1}")


if __name__ == "__main__":
    main()
