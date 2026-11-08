#!/usr/bin/env python3
"""
Standalone Logistic Regression Trainer & Evaluator
BITS Pilani WILP - M.Tech (AIML) - Machine Learning Assignment 2
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "Customer-Churn.csv")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Training Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)
    
    model_save_path = os.path.join(base_dir, "logistic_regression.joblib")
    joblib.dump(pipeline, model_save_path)
    print(f"Saved model to: {model_save_path}")
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_test_bin = (y_test == "Yes").astype(int)
    classes = list(pipeline.classes_)
    yes_idx = classes.index("Yes")
    y_proba = pipeline.predict_proba(X_test)[:, yes_idx]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test_bin, y_proba)
    prec = precision_score(y_test, y_pred, pos_label="Yes", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="Yes", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="Yes", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print("\n--- Logistic Regression Test Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()

