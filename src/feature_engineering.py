"""
Stage 3 — Feature Engineering
Scale features and apply SMOTE to balance classes.
"""
import os, pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def main():
    proc = os.path.join("data", "processed")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(proc, "X_train.csv"))
    X_val   = pd.read_csv(os.path.join(proc, "X_val.csv"))
    X_test  = pd.read_csv(os.path.join(proc, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(proc, "y_train.csv")).squeeze()

    # Scale
    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_val[NUM_COLS]   = scaler.transform(X_val[NUM_COLS])
    X_test[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # SMOTE
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

    pd.DataFrame(X_sm, columns=X_train.columns).to_csv(os.path.join(proc, "X_train_smote.csv"), index=False)
    pd.Series(y_sm, name="Churn").to_csv(os.path.join(proc, "y_train_smote.csv"), index=False)
    X_val.to_csv(os.path.join(proc, "X_val_scaled.csv"),  index=False)
    X_test.to_csv(os.path.join(proc, "X_test_scaled.csv"), index=False)

    print(f"✅ Feature engineering done | SMOTE: {len(X_sm)} samples | Scaler saved")


if __name__ == "__main__":
    main()
