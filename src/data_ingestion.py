"""
Stage 1 — Data Ingestion
Generates or downloads the raw Telco Customer Churn dataset.
"""
import os
import numpy as np
import pandas as pd

def generate_synthetic_data(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    return pd.DataFrame({
        "customerID":        [f"ID-{i:05d}" for i in range(n)],
        "gender":            np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":     np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":           np.random.choice(["Yes", "No"], n),
        "Dependents":        np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure":            np.random.randint(0, 72, n),
        "PhoneService":      np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines":     np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":   np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":    np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":       np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":          np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]),
        "PaperlessBilling":  np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod":     np.random.choice(["Electronic check", "Mailed check",
                                               "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges":    np.round(np.random.uniform(18, 120, n), 2),
        "TotalCharges":      np.round(np.random.uniform(18, 8700, n), 2),
        "Churn":             np.random.choice(["Yes", "No"], n, p=[0.265, 0.735]),
    })


def main():
    out_path = os.path.join("data", "raw", "churn_raw.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # If you have the real Kaggle CSV, place it at data/raw/ and skip generation
    if os.path.exists(out_path):
        print(f"✅ Raw data already exists: {out_path}")
        return

    df = generate_synthetic_data()
    df.to_csv(out_path, index=False)
    print(f"✅ Data saved: {out_path}  shape={df.shape}")


if __name__ == "__main__":
    main()
