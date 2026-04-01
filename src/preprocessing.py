"""
Stage 2 — Data Preprocessing
Clean, encode, and split the raw dataset.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Binary encode
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # One-hot encode
    multi_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
                  "TechSupport", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    return df


def main():
    raw_path = os.path.join("data", "raw", "churn_raw.csv")
    out_dir  = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30,
                                                         random_state=42, stratify=y)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50,
                                                         random_state=42, stratify=y_temp)

    for name, data in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                       ("y_train", y_train), ("y_val",   y_val), ("y_test",  y_test)]:
        data.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    print(f"✅ Preprocessing done | Train={X_train.shape} | Val={X_val.shape} | Test={X_test.shape}")


if __name__ == "__main__":
    main()
