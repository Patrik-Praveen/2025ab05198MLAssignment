import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


st.set_page_config(page_title="Heart Disease ML Dashboard", layout="wide")
st.title("Heart Disease Classification Dashboard")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    data = fetch_ucirepo(id=45)
    X = data.data.features
    y = data.data.targets

    df = pd.concat([X, y], axis=1)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)

    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={"num": "target"}, inplace=True)

    return df

df = load_data()

X = df.drop("target", axis=1)
y = df["target"]

# -------------------------
# Train Models If Missing
# -------------------------
os.makedirs("model", exist_ok=True)

if not os.path.exists("model/scaler.pkl"):

    st.info("Training models for first deployment...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    joblib.dump(scaler, "model/scaler.pkl")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)

        joblib.dump(model, f"model/{name}.pkl")

    st.success("Models trained successfully!")

# -------------------------
# Load Saved Models
# -------------------------
scaler = joblib.load("model/scaler.pkl")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression","Decision Tree","KNN",
     "Naive Bayes","Random Forest","XGBoost"]
)

model = joblib.load(f"model/{model_option}.pkl")

if model_option in ["Logistic Regression","KNN"]:
    X_processed = scaler.transform(X)
else:
    X_processed = X

y_pred = model.predict(X_processed)
y_prob = model.predict_proba(X_processed)[:,1]

# -------------------------
# Metrics
# -------------------------
st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(accuracy_score(y,y_pred),3))
col1.metric("Precision", round(precision_score(y,y_pred),3))

col2.metric("Recall", round(recall_score(y,y_pred),3))
col2.metric("F1 Score", round(f1_score(y,y_pred),3))

col3.metric("AUC", round(roc_auc_score(y,y_prob),3))
col3.metric("MCC", round(matthews_corrcoef(y,y_pred),3))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y,y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d')
st.pyplot(fig)

# ROC
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y,y_prob)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr)
st.pyplot(fig2)
