#!/usr/bin/env python3
"""
Model Training Script for BITS Pilani AIML Assignment 2.
Trains 5 classification models on the Telco Customer Churn dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (RBF kernel)

Evaluates all 5 models across 6 required metrics:
- Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)

Saves:
- `test_data.csv` (root directory)
- Saved pipeline models (.joblib) in `model/`
- Preprocessor artifact (.joblib) in `model/`
- Performance metrics summary in `model/metrics_summary.json`
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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


def load_and_clean_data(csv_path: str):
    """Loads and cleans the Telco Customer Churn dataset."""
    df = pd.read_csv(csv_path)
    
    # Drop customerID as it is an arbitrary identifier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # TotalCharges contains blank spaces for new customers (tenure == 0)
    # Convert to numeric, replace NaNs with 0.0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Ensure SeniorCitizen is treated as categorical string for consistent one-hot encoding
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    return df


def build_preprocessor(X: pd.DataFrame):
    """Builds a Scikit-Learn ColumnTransformer for numeric and categorical columns."""
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor, numeric_features, categorical_features


def main():
    print("=" * 60)
    print("BITS Pilani WILP - Machine Learning Assignment 2")
    print("Training & Evaluation Pipeline")
    print("=" * 60)
    
    # 1. Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Customer-Churn.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = load_and_clean_data(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    target_col = 'Churn'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    
    # 3. Save test_data.csv in the project root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_data_path = os.path.join(root_dir, 'test_data.csv')
    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_df.to_csv(test_data_path, index=False)
    print(f"Saved test_data.csv to: {test_data_path}")
    
    # 4. Build Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    
    # Save standalone preprocessor
    model_dir = os.path.dirname(__file__)
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    # Fit preprocessor on training data
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to: {preprocessor_path}")
    
    # 5. Define the 5 Classifiers
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    }
    
    # File mapping for saving models
    model_filenames = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'KNN': 'knn.joblib',
        'Naive Bayes': 'naive_bayes.joblib',
        'Support Vector Machine': 'svm.joblib'
    }
    
    # Binary numeric targets for AUC calculation ('Yes' -> 1, 'No' -> 0)
    y_test_binary = (y_test == 'Yes').astype(int)
    
    results = []
    
    print("\nTraining and evaluating models...\n")
    for name, clf in models.items():
        # Build end-to-end pipeline: Preprocessing + Classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Save pipeline model
        save_path = os.path.join(model_dir, model_filenames[name])
        joblib.dump(pipeline, save_path)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Predict probability for positive class ('Yes')
        classes = list(pipeline.classes_)
        yes_idx = classes.index('Yes')
        y_proba = pipeline.predict_proba(X_test)[:, yes_idx]
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test_binary, y_proba)
        prec = precision_score(y_test, y_pred, pos_label='Yes', zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label='Yes', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label='Yes', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results.append({
            'ML Model Name': name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4)
        })
        
        print(f"[{name}]")
        print(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    
    # 6. Summary Comparison Table
    results_df = pd.DataFrame(results)
    metrics_json_path = os.path.join(model_dir, 'metrics_summary.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 60)
    print("EVALUATION METRICS COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    print(f"\nSaved metrics summary to: {metrics_json_path}")


if __name__ == '__main__':
    main()
