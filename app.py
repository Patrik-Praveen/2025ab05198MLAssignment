
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)

st.set_page_config(page_title="Heart Disease ML Dashboard", layout="wide")

st.title("Heart Disease Classification Dashboard")
st.markdown("Predicting heart disease using multiple ML models")

@st.cache_data
def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    df = pd.concat([X, y], axis=1)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    df = df.astype(float)

    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={"num": "target"}, inplace=True)

    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop("target", axis=1)
y = df["target"]

st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression",
     "Decision Tree",
     "KNN",
     "Naive Bayes",
     "Random Forest",
     "XGBoost"]
)

scaler = joblib.load("model/scaler.pkl")

model_files = {
    "Logistic Regression": "model/Logistic Regression.pkl",
    "Decision Tree": "model/Decision Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive Bayes.pkl",
    "Random Forest": "model/Random Forest.pkl",
    "XGBoost": "model/XGBoost.pkl"
}

model = joblib.load(model_files[model_option])

if model_option in ["Logistic Regression", "KNN"]:
    X_processed = scaler.transform(X)
else:
    X_processed = X

y_pred = model.predict(X_processed)
y_prob = model.predict_proba(X_processed)[:, 1]

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
mcc = matthews_corrcoef(y, y_pred)

st.subheader("Model Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(accuracy, 3))
col1.metric("Precision", round(precision, 3))

col2.metric("Recall", round(recall, 3))
col2.metric("F1 Score", round(f1, 3))

col3.metric("AUC Score", round(auc, 3))
col3.metric("MCC Score", round(mcc, 3))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
st.pyplot(fig)

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y, y_prob)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
st.pyplot(fig2)

if model_option == "Random Forest":
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))
