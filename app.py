import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="SecurePay - Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,58,237,0.1));
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0,212,255,0.3);
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/test.csv")
    X_test = df.drop("Class", axis=1)
    y_test = df["Class"]
    return X_test, y_test, df

@st.cache_resource
def load_models():
    rf_model = joblib.load("artifacts/models/rf_model.pkl")
    xgb_model = joblib.load("artifacts/models/xgb_model.pkl")
    return rf_model, xgb_model

X_test, y_test, df = load_data()
rf_model, xgb_model = load_models()


st.sidebar.markdown("## 🎛️ Controls")
model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["Random Forest", "XGBoost"],
    help="Select which model to analyze"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.info(f"""
**Total Transactions:** {len(df):,}  
**Fraud Cases:** {y_test.sum():,}  
**Fraud Rate:** {(y_test.sum()/len(y_test)*100):.3f}%  
**Features:** {X_test.shape[1]}
""")


model = rf_model if model_choice == "Random Forest" else xgb_model

with st.spinner(f"Running {model_choice} predictions..."):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

st.markdown('<h1 class="main-header">🛡️ SecurePay - Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center; color: #00D4FF;'>Model: {model_choice}</h2>", unsafe_allow_html=True)

st.markdown("### Model Performance Metrics 📈")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Accuracy", 
        value=f"{acc:.3f}",
        delta=f"{(acc-0.5)*100:.1f}%" if acc > 0.5 else None
    )

with col2:
    st.metric(
        label="Precision", 
        value=f"{prec:.3f}",
        help="True Positives / (True Positives + False Positives)"
    )

with col3:
    st.metric(
        label="Recall", 
        value=f"{rec:.3f}",
        help="True Positives / (True Positives + False Negatives)"
    )

with col4:
    st.metric(
        label="F1-Score", 
        value=f"{f1:.3f}",
        help="2 * (Precision * Recall) / (Precision + Recall)"
    )

with col5:
    st.metric(
        label="ROC AUC", 
        value=f"{auc:.3f}",
        help="Area Under the ROC Curve"
    )

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=["Non-Fraud", "Fraud"], 
        yticklabels=["Non-Fraud", "Fraud"], 
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax.set_ylabel("Actual", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_choice} - Confusion Matrix", fontsize=14, fontweight='bold')
    
    st.pyplot(fig)

with col_right:
    st.markdown("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=3, label=f"{model_choice} (AUC = {auc:.3f})", color='#00D4FF')
    ax.plot([0, 1], [0, 1], '--', color="red", alpha=0.7, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.3, color='#00D4FF')
    
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_choice} - ROC Curve", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

st.markdown("---")

st.markdown("### Detailed Analysis 🔍")

col_analysis1, col_analysis2 = st.columns(2)

with col_analysis1:
    st.markdown("#### Prediction Score Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fraud_scores = y_proba[y_test == 1]
    non_fraud_scores = y_proba[y_test == 0]
    
    ax.hist(non_fraud_scores, bins=50, alpha=0.7, label='Non-Fraud', color='lightblue', density=True)
    ax.hist(fraud_scores, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    
    ax.set_xlabel('Prediction Score', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Distribution of Prediction Scores', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with col_analysis2:
    st.markdown("#### Threshold Analysis")
    
    thresholds_range = np.arange(0.1, 1.0, 0.1)
    threshold_metrics = []
    
    for threshold in thresholds_range:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        precision_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        threshold_metrics.append({
            'Threshold': threshold,
            'Precision': precision_thresh,
            'Recall': recall_thresh,
            'F1-Score': f1_thresh
        })
    
    threshold_df = pd.DataFrame(threshold_metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_df['Threshold'], threshold_df['Precision'], 'o-', label='Precision', linewidth=2)
    ax.plot(threshold_df['Threshold'], threshold_df['Recall'], 's-', label='Recall', linewidth=2)
    ax.plot(threshold_df['Threshold'], threshold_df['F1-Score'], '^-', label='F1-Score', linewidth=2)
    
    ax.set_xlabel('Threshold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Metrics vs Threshold', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    st.pyplot(fig)

st.markdown("---")
st.markdown("### Model Comparison")

if st.button("Compare Both Models", type="primary"):
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred),
        'F1-Score': f1_score(y_test, rf_pred),
        'ROC AUC': roc_auc_score(y_test, rf_proba)
    }
    
    xgb_metrics = {
        'Model': 'XGBoost',
        'Accuracy': accuracy_score(y_test, xgb_pred),
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'F1-Score': f1_score(y_test, xgb_pred),
        'ROC AUC': roc_auc_score(y_test, xgb_proba)
    }
    
    comparison_df = pd.DataFrame([rf_metrics, xgb_metrics])
    
    st.markdown("#### Side-by-Side Comparison")
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']),
        use_container_width=True
    )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🛡️ <strong>SecurePay</strong> - Advanced Credit Card Fraud Detection Platform</p>
    <p>Powered by Machine Learning • Protecting Financial Transactions Worldwide</p>
</div>
""", unsafe_allow_html=True)