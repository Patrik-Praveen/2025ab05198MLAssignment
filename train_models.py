
import os
import pandas as pd
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal","target"]

try:
    cleveland = pd.read_csv("processed.cleveland.data", names=columns)
    hungarian = pd.read_csv("processed.hungarian.data", names=columns)
    switzerland = pd.read_csv("processed.switzerland.data", names=columns)
    va = pd.read_csv("processed.va.data", names=columns)

    df = pd.concat([cleveland, hungarian, switzerland, va], ignore_index=True)
except FileNotFoundError:
    # If local raw files aren't present, try fetching the dataset from the UCI
    # repository using ucimlrepo (installed via requirements.txt).
    try:
        from ucimlrepo.fetch import fetch_ucirepo
    except Exception:
        raise

    print("Local data files not found — downloading dataset from UCI repository...")
    result = fetch_ucirepo(name="Heart Disease")
    # fetch_ucirepo returns a dotdict with data.original containing the full dataframe
    df = result.data.original

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)
# accommodate datasets where the target column is named 'num' (UCI naming)
if 'target' not in df.columns and 'num' in df.columns:
    df['target'] = df['num']
if 'target' not in df.columns:
    raise KeyError("Could not find target column in dataset (expected 'target' or 'num')")
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")

print("Models trained and saved successfully.")
