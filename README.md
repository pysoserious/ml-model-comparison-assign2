# ML Model Comparison & Deployment | BITS Pilani AIML Assignment 2

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-model-comparison-assign2.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete implementation for **Assignment - 2** of the **Machine Learning** course in the **M.Tech (AIML)** program at **BITS Pilani (Work Integrated Learning Programmes Division)**.

---

## a. Problem Statement

Customer churn is a critical business metric in subscription-based industries such as telecommunications. Retaining existing customers is significantly more cost-effective than acquiring new ones. The goal of this project is to build, compare, and deploy machine learning classification models to predict whether a telecom customer is likely to churn (`Churn = 'Yes'`) or remain with the company (`Churn = 'No'`).

By evaluating five distinct classification paradigms under rigorous out-of-sample testing, this project identifies the optimal algorithm for churn risk mitigation and provides an interactive, production-ready Streamlit web application for real-time model evaluation and single-customer inference.

---

## b. Dataset Description

- **Dataset Name**: Telco Customer Churn (IBM / Kaggle Public Repository)
- **Problem Type**: Binary Classification (`Churn`: `'Yes'` / `'No'`)
- **Total Instances**: **7,043** records *(Mandatory requirement $\ge 500$ instances — MET)*
- **Total Features**: **19** input features + 1 target column *(Mandatory requirement $\ge 12$ features — MET)*
- **Target Distribution**:
  - `No` (Retained): **5,174** instances (~**73.46%**)
  - `Yes` (Churned): **1,869** instances (~**26.54%**)

### Feature Breakdown

| Category | Features | Description |
| :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer demographic background and household status. |
| **Services Subscribed** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Subscribed voice, broadband, security, and entertainment services. |
| **Account Information** | `Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `TotalCharges` | Contractual terms, billing preferences, tenure duration, and financial charges. |
| **Target Variable** | `Churn` | Whether the customer churned within the last month (`Yes` or `No`). |

### Preprocessing & Hygiene Pipeline
1. **Identifier Removal**: `customerID` was removed as it is an arbitrary non-predictive primary key.
2. **Missing / Whitespace Value Handling**: `TotalCharges` contained 11 blank space entries (`" "`) corresponding to new customers with `tenure == 0`. These were converted to numeric values and imputed to `0.0`.
3. **Encoding & Scaling**:
   - Numerical attributes (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler`.
   - Categorical attributes were transformed using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
4. **Stratified Split**: The dataset was partitioned into **80% training** (5,634 rows) and **20% test** (1,409 rows) stratified on the target column `Churn` to preserve identical class ratios across splits. The test split is exported as `test_data.csv` for independent evaluation.

---

## c. GitHub Repository Link

- **Repository URL**: [https://github.com/pysoserious/ml-model-comparison-assign2](https://github.com/pysoserious/ml-model-comparison-assign2)

### Repository Structure
```text
ml-model-comparison-assign2/
│-- app.py                     # Interactive Streamlit Web Application
│-- requirements.txt           # Python dependencies for local & cloud deployment
│-- README.md                  # Comprehensive documentation and comparison report
│-- test_data.csv              # Unseen 20% test split (1,409 rows) with target column
│-- data/
│   └── Customer-Churn.csv     # Full source dataset
│-- model/
│   │-- train_models.py        # Complete training, evaluation, and artifact export script
│   │-- logistic_regression.py # Standalone runner for Logistic Regression
│   │-- model_training.ipynb   # Jupyter Notebook with EDA, training, and visualizations
│   │-- preprocessor.joblib    # Fitted ColumnTransformer preprocessing artifact
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- svm.joblib
│   └── metrics_summary.json   # Machine-readable evaluation metrics summary
```

---

## d. Models Used & Evaluation Metrics

All five classification models were trained on the identical 80% training set and evaluated on the out-of-sample `test_data.csv` (1,409 instances) across six mandatory performance metrics:
1. **Accuracy**
2. **AUC Score** (Area Under the ROC Curve)
3. **Precision** (Positive Class: `Churn = 'Yes'`)
4. **Recall** (Positive Class: `Churn = 'Yes'`)
5. **F1 Score** (Harmonic mean of Precision and Recall)
6. **Matthews Correlation Coefficient (MCC)**

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8055** | **0.8420** | **0.6572** | 0.5588 | **0.6040** | **0.4790** |
| **Decision Tree** | 0.7984 | 0.8276 | 0.6347 | 0.5668 | 0.5989 | 0.4662 |
| **KNN** | 0.7637 | 0.8024 | 0.5538 | 0.5642 | 0.5589 | 0.3976 |
| **Naive Bayes** | 0.6955 | 0.8075 | 0.4597 | **0.8396** | 0.5941 | 0.4268 |
| **Support Vector Machine** | 0.7899 | 0.7928 | 0.6373 | 0.4840 | 0.5502 | 0.4232 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as the top-performing model overall. Achieves the highest Accuracy (**80.55%**), highest AUC (**0.8420**), highest Precision (**65.72%**), highest F1 Score (**0.6040**), and highest MCC (**0.4790**). The log-odds linear formulation fits the standardized continuous features and one-hot encoded categories well, providing smoothly calibrated probabilities essential for risk tiering. |
| **Decision Tree** | Regularized with `max_depth=5` and `min_samples_leaf=10` to prevent overfitting. Delivers strong interpretability and competitive accuracy (**79.84%**) and AUC (**0.8276**). The top decision nodes indicate that month-to-month contracts and high monthly charges are the primary drivers of churn. |
| **KNN** | Achieves moderate performance (Accuracy **76.37%**, AUC **0.8024**, MCC **0.3976**). Suffers slightly from the curse of dimensionality in the one-hot encoded categorical space, which distorts Euclidean distance metrics between sparse binary vectors. |
| **Naive Bayes** | Achieves the highest Recall (**83.96%**), successfully flagging almost 84% of all churning customers. However, due to its strict assumption of conditional feature independence (which is violated by correlations such as tenure vs. TotalCharges), it produces the lowest Precision (**45.97%**) and lowest Accuracy (**69.55%**). Suitable in scenarios where missing a churner carries a severe business penalty. |
| **Support Vector Machine** | Utilizing an RBF kernel with probability calibration, SVM achieves **78.99%** Accuracy and **0.7928** AUC. While it establishes a robust non-linear decision boundary, it exhibits a more conservative recall (**48.40%**) on the minority churn class compared to Logistic Regression and Decision Trees. |

### Overall, Winner for your dataset?

> **Overall Winner: Logistic Regression**
> 
> **Justification:**  
> Logistic Regression is selected as the definitive winner for this telecommunications churn problem based on the following technical and operational criteria:
> 1. **Metric Dominance**: It achieves the highest score across 5 out of 6 evaluation metrics — Accuracy (**0.8055**), ROC-AUC (**0.8420**), Precision (**0.6572**), F1 Score (**0.6040**), and Matthews Correlation Coefficient (**0.4790**).
> 2. **Resistance to Imbalance**: In an imbalanced setting (26.5% churn rate), Accuracy alone can be misleading. Logistic Regression's superior MCC (0.4790) and AUC (0.8420) confirm genuine predictive power across both classes without bias.
> 3. **Business Actionability**: Logistic Regression outputs well-calibrated, monotonic probability scores. In practical retention operations, telecom managers do not simply act on binary flags; they rank customers by predicted churn risk and allocate retention budgets (discounts, outreach) to the highest-risk tiers.
> 4. **Inference Efficiency**: Requires minimal computational overhead and memory in production, ensuring near-instant predictions on Streamlit Community Cloud.

---

## f. Streamlit Web Application Features

The interactive web application (`app.py`) provides:
1. **Dataset Upload Option (CSV)**: Users can upload any test CSV containing feature columns and the `Churn` target. If no file is provided, it automatically defaults to `test_data.csv` (1,409 rows) for immediate demonstration within Streamlit free tier limits.
2. **Model Selection Dropdown**: Easily switch between **Logistic Regression**, **Decision Tree**, **KNN**, **Naive Bayes**, **Support Vector Machine**, or view the **All Models Comparative Dashboard**.
3. **Display of Evaluation Metrics**: Dynamic KPI cards displaying all 6 metrics (**Accuracy, AUC, Precision, Recall, F1, MCC**) computed without retraining.
4. **Confusion Matrix & Classification Report**: Annotated Seaborn confusion matrix heatmaps and complete classification report tables.
5. **ROC Curves & Comparative Charts**: Interactive multi-model ROC curves and comparative multi-metric bar charts.
6. **Live Single-Customer Predictor**: Form with sliders and dropdowns allowing evaluators to simulate customer profiles and predict real-time churn risk.

---

## g. How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/pysoserious/ml-model-comparison-assign2.git
cd ml-model-comparison-assign2
```

### 2. Set up virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train models (Optional - pre-trained models are already saved in `model/`)
```bash
python model/train_models.py
```

### 4. Launch the Streamlit application
```bash
streamlit run app.py
```
The app will automatically launch in your browser at `http://localhost:8501`.

---

## h. Deployment on Streamlit Community Cloud

1. Push all code and artifacts to GitHub repository `pysoserious/ml-model-comparison-assign2` on branch `main`.
2. Navigate to [https://share.streamlit.io/](https://share.streamlit.io/).
3. Connect your GitHub account and click **"New App"**.
4. Select repository: `pysoserious/ml-model-comparison-assign2`, Branch: `main`, Main file path: `app.py`.
5. Click **"Deploy"**.
