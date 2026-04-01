"""
Training Pipeline Module
=========================
Trains models, logs to MLflow, saves best model artifact.
"""
import os
import pickle
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def get_models() -> dict:
    return {
        "LogisticRegression": {
            "model":  LogisticRegression(max_iter=500, random_state=42),
            "params": {"max_iter": 500, "solver": "lbfgs"},
        },
        "RandomForest": {
            "model":  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 200, "max_depth": 10},
        },
        "GradientBoosting": {
            "model":  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            "params": {"n_estimators": 200, "lr": 0.05, "max_depth": 5},
        },
    }


def eval_metrics(model, X, y, split="Val") -> dict:
    p  = model.predict(X)
    pr = model.predict_proba(X)[:, 1]
    return {
        f"{split}_Accuracy":  round(accuracy_score(y, p), 4),
        f"{split}_Precision": round(precision_score(y, p), 4),
        f"{split}_Recall":    round(recall_score(y, p), 4),
        f"{split}_F1":        round(f1_score(y, p), 4),
        f"{split}_ROC_AUC":   round(roc_auc_score(y, pr), 4),
    }


def run_training():
    """Full training pipeline with MLflow tracking."""
    mlflow.set_tracking_uri(f"file:{os.path.join(os.path.dirname(__file__),'..','mlruns')}")
    mlflow.set_experiment("Customer_Churn_Prediction")

    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_smote.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_smote.csv")).squeeze()
    X_val   = pd.read_csv(os.path.join(DATA_DIR, "X_val_scaled.csv"))
    y_val   = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()

    best_model, best_name, best_f1 = None, "", 0.0

    for name, cfg in get_models().items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            m = cfg["model"]
            m.fit(X_train, y_train)
            mlflow.log_params(cfg["params"])
            val_m = eval_metrics(m, X_val, y_val, "Val")
            mlflow.log_metrics({**eval_metrics(m, X_train, y_train, "Train"), **val_m})
            mlflow.sklearn.log_model(m, "model", registered_model_name=f"Churn_{name}")
            logger.info(f"  F1={val_m['Val_F1']} | AUC={val_m['Val_ROC_AUC']}")
            if val_m["Val_F1"] > best_f1:
                best_f1, best_name, best_model = val_m["Val_F1"], name, m

    # Save best
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(list(X_train.columns), f)

    logger.info(f"🏆 Best: {best_name} | F1={best_f1}")
    return best_model, best_name


if __name__ == "__main__":
    run_training()
