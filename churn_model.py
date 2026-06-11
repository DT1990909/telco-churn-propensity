"""
Telco Customer Churn Propensity Model
=====================================
Predicts customer churn using logistic regression and random forest,
evaluates with ROC-AUC, and translates results into plain-English
audience segment insights for non-technical stakeholders.

Usage:
    python churn_model.py                  # runs on bundled synthetic data
    python churn_model.py path/to/telco.csv  # runs on the Kaggle Telco dataset

Author: Daryl | Research & Data Analyst
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def generate_synthetic_telco(n=5000):
    """Generate a realistic synthetic telco dataset so the project runs
    out-of-the-box. Mirrors the structure of the Kaggle Telco dataset."""
    df = pd.DataFrame({
        "tenure": np.random.exponential(24, n).clip(0, 72).round(),
        "MonthlyCharges": np.random.normal(65, 30, n).clip(18, 120).round(2),
        "Contract": np.random.choice(
            ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "InternetService": np.random.choice(
            ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "TechSupport": np.random.choice(["Yes", "No"], n, p=[0.29, 0.71]),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            n, p=[0.34, 0.23, 0.22, 0.21]),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
    })
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]
                          * np.random.uniform(0.9, 1.1, n)).round(2)

    # Churn probability driven by realistic relationships
    logit = (
        -2.6
        - 0.045 * df["tenure"]
        + 0.012 * df["MonthlyCharges"]
        + 1.1 * (df["Contract"] == "Month-to-month")
        + 0.6 * (df["InternetService"] == "Fiber optic")
        + 0.5 * (df["TechSupport"] == "No")
        + 0.4 * (df["PaymentMethod"] == "Electronic check")
        + 0.3 * df["SeniorCitizen"]
    )
    p = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.random.binomial(1, p)
    return df


def load_data():
    if len(sys.argv) > 1:
        print(f"Loading dataset: {sys.argv[1]}")
        df = pd.read_csv(sys.argv[1])
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])
        df["Churn"] = (df["Churn"] == "Yes").astype(int)
        keep = ["tenure", "MonthlyCharges", "TotalCharges", "Contract",
                "InternetService", "TechSupport", "PaymentMethod",
                "SeniorCitizen", "Churn"]
        return df[keep]
    print("No dataset supplied - using bundled synthetic data.")
    return generate_synthetic_telco()


# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
def preprocess(df):
    X = pd.get_dummies(df.drop(columns="Churn"), drop_first=True)
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.25,
                            stratify=y, random_state=RANDOM_STATE)


# ----------------------------------------------------------------------
# 3. Modelling
# ----------------------------------------------------------------------
def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_s, y_train)

    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    results = {}
    for name, model, X_te in [("Logistic Regression", logreg, X_test_s),
                              ("Random Forest", rf, X_test)]:
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results[name] = {"model": model, "proba": proba, "auc": auc}
        print(f"\n{name}  |  ROC-AUC: {auc:.3f}")
        print(classification_report(y_test, (proba >= 0.5).astype(int),
                                    target_names=["Stayed", "Churned"]))
    return results, rf, X_train.columns


# ----------------------------------------------------------------------
# 4. Visualisation
# ----------------------------------------------------------------------
def plot_roc(results, y_test):
    plt.figure(figsize=(7, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Churn Model ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    print("\nSaved: roc_curves.png")


def plot_feature_importance(rf, feature_names):
    imp = (pd.Series(rf.feature_importances_, index=feature_names)
           .sort_values().tail(10))
    plt.figure(figsize=(8, 5))
    imp.plot(kind="barh", color="#2c5f8a")
    plt.title("Top 10 Churn Drivers (Random Forest)")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved: feature_importance.png")
    return imp


# ----------------------------------------------------------------------
# 5. Stakeholder insights
# ----------------------------------------------------------------------
def segment_insights(df):
    print("\n" + "=" * 60)
    print("AUDIENCE SEGMENT INSIGHTS (plain English)")
    print("=" * 60)
    seg = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    for contract, rate in seg.items():
        print(f"  - {contract} customers churn at {rate:.0%}")
    new = df[df["tenure"] <= 6]["Churn"].mean()
    old = df[df["tenure"] > 24]["Churn"].mean()
    print(f"  - Customers in their first 6 months churn at {new:.0%}, "
          f"vs {old:.0%} for customers past 2 years.")
    print("  - Retention budget is best aimed at new, month-to-month,")
    print("    fiber-optic customers without tech support.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print(f"Dataset: {len(df):,} customers | churn rate: {df['Churn'].mean():.1%}")
    X_train, X_test, y_train, y_test = preprocess(df)
    results, rf, feats = train_models(X_train, X_test, y_train, y_test)
    plot_roc(results, y_test)
    plot_feature_importance(rf, feats)
    segment_insights(df)
