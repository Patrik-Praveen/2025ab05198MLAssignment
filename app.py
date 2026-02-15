import streamlit as st
import pandas as pd
import numpy as np
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


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Heart Disease ML Dashboard", layout="wide")
st.title("❤️ Heart Disease Classification Dashboard")
st.markdown("Production-Safe Deployment Version")


# -------------------------------------------------
# Load Dataset (Cleveland dataset from UCI)
# -------------------------------------------------
@st.cache_data
def load_data():
    data = fetch_ucirepo(id=45)
    X = data.data.features
    y = data.data.targets

    df = pd.concat([X, y], axis=1)

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)

    # Convert multi-class to binary
    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={"num": "target"}, inplace=True)

    return df


df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Prepare Features
# -------------------------------------------------
expected_columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

X = df[expected_columns].copy()
y = df["target"].copy()

# Convert to numpy (avoids sklearn feature name errors)
X = X.values


# -------------------------------------------------
# Train/Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (only for LR and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------
# Model Selection
# -------------------------------------------------
st.sidebar.header("⚙ Model Selection")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression",
     "Decision Tree",
     "KNN",
     "Naive Bayes",
     "Random Forest",
     "XGBoost"]
)


# -------------------------------------------------
# Train Selected Model
# -------------------------------------------------
if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

elif mod
