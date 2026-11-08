"""
Customer Churn Prediction & Model Evaluation Dashboard
BITS Pilani WILP - M.Tech (AIML) - Machine Learning Assignment 2

Features:
- Dataset upload option (CSV) with automatic fallback to test_data.csv
- Model selection dropdown (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, All Models)
- Dynamic computation and display of 6 key evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Visual Confusion Matrix heatmap and formatted Classification Report
- ROC Curves and multi-model comparative visualizations
- Interactive single-customer prediction tool for real-time inference
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Model Comparison Dashboard | BITS AIML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Paths & Model Registry
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST_DATA = os.path.join(BASE_DIR, "test_data.csv")

MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_saved_model(filename: str):
    """Loads a joblib pipeline model from the model directory."""
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        return None
    return joblib.load(filepath)


@st.cache_resource
def load_all_models():
    """Preloads all 5 classification pipelines."""
    models = {}
    for name, fname in MODEL_REGISTRY.items():
        m = load_saved_model(fname)
        if m is not None:
            models[name] = m
    return models


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses uploaded or default test dataframe."""
    df_clean = df.copy()
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
    
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'].astype(str).str.strip(), errors='coerce'
        ).fillna(0.0)
        
    if 'SeniorCitizen' in df_clean.columns:
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(str)
        
    return df_clean


def compute_metrics(y_true, y_pred, y_proba):
    """Computes the 6 mandated evaluation metrics."""
    y_true_bin = (y_true == 'Yes').astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true_bin, y_proba) if y_proba is not None else 0.0
    prec = precision_score(y_true, y_pred, pos_label='Yes', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='Yes', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='Yes', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    }


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("🛠️ Configuration")

# 1. Dataset Upload Option
st.sidebar.subheader("1. Test Dataset")

# Provide download sample button for evaluators
if os.path.exists(DEFAULT_TEST_DATA):
    with open(DEFAULT_TEST_DATA, "rb") as f_sample:
        st.sidebar.download_button(
            label="📥 Download Sample Test CSV",
            data=f_sample.read(),
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download a benchmark test CSV to test the upload feature."
        )

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"],
    help="Upload a CSV with input features and the 'Churn' target column. Streamlit free tier has limited capacity, so upload test data only."
)

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded: `{uploaded_file.name}` ({raw_df.shape[0]} rows)")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        raw_df = None
else:
    if os.path.exists(DEFAULT_TEST_DATA):
        raw_df = pd.read_csv(DEFAULT_TEST_DATA)
        st.sidebar.info(f"Loaded benchmark `test_data.csv` ({raw_df.shape[0]} rows)")
    else:
        st.sidebar.warning("Default `test_data.csv` not found. Please upload a test CSV.")
        raw_df = None

# 2. Model Selection Dropdown
st.sidebar.subheader("2. Model Selection")
model_options = list(MODEL_REGISTRY.keys()) + ["All Models (Comparative View)"]
selected_option = st.sidebar.selectbox(
    "Choose Classifier",
    options=model_options,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Assignment Information**
    - **Course**: BITS Pilani WILP M.Tech (AIML)
    - **Subject**: Machine Learning (Assignment 2)
    - **Domain**: Telco Customer Churn Prediction
    """
)


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">📈 ML Model Comparison & Evaluation Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Comparative performance analysis of 5 classification models on the Telco Customer Churn dataset with out-of-sample evaluation.</div>',
    unsafe_allow_html=True
)

if raw_df is None:
    st.error("⚠️ No dataset available. Please upload a valid test CSV file to proceed.")
    st.stop()

# Validate Churn target column
if 'Churn' not in raw_df.columns:
    st.error("⚠️ The uploaded CSV must contain the target column `'Churn'` with values `'Yes'` and `'No'`.")
    st.write("Current columns detected:", list(raw_df.columns))
    st.stop()

REQUIRED_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]
missing_features = [c for c in REQUIRED_FEATURES if c not in raw_df.columns]
if missing_features:
    st.error(f"⚠️ The uploaded CSV is missing the following required feature column(s): `{missing_features}`")
    st.info("Please ensure your uploaded test dataset contains all 19 customer features and the 'Churn' target.")
    st.stop()

# Prepare Data
df_eval = clean_dataframe(raw_df)
X_test = df_eval.drop(columns=['Churn'])
y_test = df_eval['Churn']

# Load Models
all_models = load_all_models()
if not all_models:
    st.error("⚠️ Saved model artifacts (.joblib) were not found in the `model/` folder. Please ensure models are trained.")
    st.stop()


# ---------------------------------------------------------
# Single Model View or Comparative View
# ---------------------------------------------------------
if selected_option != "All Models (Comparative View)":
    current_model_name = selected_option
    pipeline = all_models.get(current_model_name)
    
    if pipeline is None:
        st.error(f"Could not load artifact for {current_model_name}.")
        st.stop()
        
    # Predictions
    with st.spinner(f"Evaluating {current_model_name}..."):
        y_pred = pipeline.predict(X_test)
        
        classes = list(pipeline.classes_)
        yes_idx = classes.index('Yes') if 'Yes' in classes else 1
        y_proba = pipeline.predict_proba(X_test)[:, yes_idx]
        
        metrics = compute_metrics(y_test, y_pred, y_proba)
        
    st.subheader(f"🔍 Model Evaluation: {current_model_name}")
    
    # 6 Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["Accuracy"]:.4f}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["AUC"]:.4f}</div><div class="metric-label">AUC Score</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["Precision"]:.4f}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["Recall"]:.4f}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["F1"]:.4f}</div><div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["MCC"]:.4f}</div><div class="metric-label">MCC Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix & Report", "📈 ROC Curve Analysis", "👤 Live Single Prediction"])
    
    with tab1:
        col_cm, col_cr = st.columns([1, 1])
        
        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Stayed (No)', 'Churned (Yes)'],
                yticklabels=['Stayed (No)', 'Churned (Yes)'],
                annot_kws={"size": 13, "weight": "bold"},
                ax=ax
            )
            ax.set_xlabel("Predicted Label", fontweight="bold")
            ax.set_ylabel("True Label", fontweight="bold")
            ax.set_title(f"Confusion Matrix - {current_model_name}", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)
            
        with col_cr:
            st.markdown("#### Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(
                report_df.style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                }).background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
                use_container_width=True
            )
            
            st.caption(
                f"Evaluation performed on **{len(y_test)}** test instances without retraining."
            )
            
    with tab2:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
        y_test_bin = (y_test == 'Yes').astype(int)
        fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
        
        fig_roc, ax_roc = plt.subplots(figsize=(7, 4.5))
        ax_roc.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f'{current_model_name} (AUC = {metrics["AUC"]:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate (Recall)', fontweight='bold')
        ax_roc.set_title(f'ROC Curve - {current_model_name}', fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_roc)
        plt.close(fig_roc)
        
    with tab3:
        st.markdown("#### Test Real-Time Single-Customer Prediction")
        st.write("Tweak customer attributes below to predict whether this customer is likely to churn:")
        
        with st.form("single_predict_form"):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                in_gender = st.selectbox("Gender", ["Female", "Male"])
                in_senior = st.selectbox("Senior Citizen", ["0", "1"])
                in_partner = st.selectbox("Partner", ["Yes", "No"])
                in_dependents = st.selectbox("Dependents", ["No", "Yes"])
                in_tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
                
            with col_f2:
                in_phone = st.selectbox("Phone Service", ["Yes", "No"])
                in_multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                in_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                in_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                in_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                in_device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

            with col_f3:
                in_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                in_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                in_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                in_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                in_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                in_payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                in_monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0, step=0.5)
                in_total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(in_tenure * in_monthly), step=10.0)

            submitted = st.form_submit_button("Predict Churn Risk", type="primary")
            
            if submitted:
                input_dict = {
                    'gender': in_gender,
                    'SeniorCitizen': in_senior,
                    'Partner': in_partner,
                    'Dependents': in_dependents,
                    'tenure': in_tenure,
                    'PhoneService': in_phone,
                    'MultipleLines': in_multiple,
                    'InternetService': in_internet,
                    'OnlineSecurity': in_security,
                    'OnlineBackup': in_backup,
                    'DeviceProtection': in_device,
                    'TechSupport': in_tech,
                    'StreamingTV': in_tv,
                    'StreamingMovies': in_movies,
                    'Contract': in_contract,
                    'PaperlessBilling': in_paperless,
                    'PaymentMethod': in_payment,
                    'MonthlyCharges': in_monthly,
                    'TotalCharges': in_total
                }
                sample_df = pd.DataFrame([input_dict])
                sample_pred = pipeline.predict(sample_df)[0]
                sample_proba = pipeline.predict_proba(sample_df)[:, yes_idx][0]
                
                if sample_pred == 'Yes':
                    st.error(f"🚨 **High Risk of Churn!** (Predicted Churn Probability: **{sample_proba * 100:.2f}%**)")
                else:
                    st.success(f"✅ **Customer Retained!** (Predicted Churn Probability: **{sample_proba * 100:.2f}%**)")

else:
    # ---------------------------------------------------------
    # Comparative View (All 5 Models)
    # ---------------------------------------------------------
    st.subheader("📊 Comparative Evaluation: All 5 Mandated Classifiers")
    st.write(
        f"Evaluating all models simultaneously on the test dataset (**{len(y_test)}** records) without retraining:"
    )
    
    comparative_results = []
    roc_data = {}
    
    y_test_bin = (y_test == 'Yes').astype(int)
    
    for name, pipeline in all_models.items():
        y_pred = pipeline.predict(X_test)
        classes = list(pipeline.classes_)
        yes_idx = classes.index('Yes') if 'Yes' in classes else 1
        y_proba = pipeline.predict_proba(X_test)[:, yes_idx]
        
        m = compute_metrics(y_test, y_pred, y_proba)
        comparative_results.append({
            "ML Model Name": name,
            "Accuracy": round(m["Accuracy"], 4),
            "AUC": round(m["AUC"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "MCC": round(m["MCC"], 4)
        })
        
        fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
        roc_data[name] = (fpr, tpr, m["AUC"])
        
    comp_df = pd.DataFrame(comparative_results)
    
    # Required Comparison Table format matching Section 3 - Step 5
    st.markdown("#### 1. Evaluation Metrics Comparison Table")
    st.dataframe(
        comp_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}"
        }).highlight_max(axis=0, color="#BBF7D0", subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]),
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Comparisons: Multi-metric bar chart & Multi-ROC
    col_chart, col_roc_all = st.columns([1.1, 0.9])
    
    with col_chart:
        st.markdown("#### 2. Metric Comparison Bar Chart")
        plot_df = comp_df.melt(id_vars=["ML Model Name"], var_name="Metric", value_name="Score")
        
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4.8))
        sns.barplot(data=plot_df, x="Metric", y="Score", hue="ML Model Name", palette="tab10", ax=ax_bar)
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_ylabel("Score", fontweight="bold")
        ax_bar.set_xlabel("Evaluation Metric", fontweight="bold")
        ax_bar.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax_bar.grid(axis='y', linestyle=':', alpha=0.7)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
        
    with col_roc_all:
        st.markdown("#### 3. Combined ROC Curves")
        fig_multiroc, ax_mroc = plt.subplots(figsize=(6, 4.8))
        
        colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
        for (m_name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
            ax_mroc.plot(fpr, tpr, lw=2, color=color, label=f"{m_name} ({auc_val:.3f})")
            
        ax_mroc.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.5, linestyle='--', label='Chance (0.500)')
        ax_mroc.set_xlim([0.0, 1.0])
        ax_mroc.set_ylim([0.0, 1.05])
        ax_mroc.set_xlabel('False Positive Rate', fontweight='bold')
        ax_mroc.set_ylabel('True Positive Rate', fontweight='bold')
        ax_mroc.legend(loc='lower right', fontsize=8.5)
        ax_mroc.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_multiroc)
        plt.close(fig_multiroc)

    # Winner & Observations Summary
    st.markdown("---")
    st.subheader("🏆 Model Performance Summary & Winner Selection")
    
    obs_data = [
        {
            "ML Model Name": "Logistic Regression",
            "Key Observations": "Strong, well-calibrated baseline. High AUC (~0.84) and balanced precision/recall trade-off. Highly interpretable."
        },
        {
            "ML Model Name": "Decision Tree",
            "Key Observations": "Constrained with max_depth=5 to prevent overfitting. Clear decision rules, but lower AUC and MCC compared to linear/SVM models."
        },
        {
            "ML Model Name": "KNN",
            "Key Observations": "Sensitive to feature distance and density. Performs moderately well but computation scales with dataset size."
        },
        {
            "ML Model Name": "Naive Bayes",
            "Key Observations": "Assumes feature independence. Produces highest Recall for churners (captures churn risk aggressively) at the expense of Precision."
        },
        {
            "ML Model Name": "Support Vector Machine",
            "Key Observations": "RBF kernel captures non-linear boundary effectively. Achieves competitive Accuracy and strong AUC (~0.83), closely matching Logistic Regression."
        }
    ]
    st.table(pd.DataFrame(obs_data))
    
    st.info(
        """
        **Overall Winner for Telco Customer Churn:** **Logistic Regression**  
        *Justification:* Logistic Regression achieves the highest ROC-AUC score (~0.844) and the highest Matthews Correlation Coefficient (MCC ~0.47) while maintaining balanced precision and recall. In telecommunications churn management, a well-calibrated probability score allows businesses to tier customer retention interventions effectively.
        """
    )


# ---------------------------------------------------------
# Dataset Preview
# ---------------------------------------------------------
with st.expander("🔍 Inspect Uploaded Test Data"):
    st.write(f"Total Rows: **{raw_df.shape[0]}** | Total Columns: **{raw_df.shape[1]}**")
    st.dataframe(raw_df.head(10), use_container_width=True)

