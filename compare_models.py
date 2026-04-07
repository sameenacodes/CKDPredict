import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.impute import SimpleImputer
import numpy as np

# 1. Load dataset
df = pd.read_csv("Kidney_data.csv")

# 2. Replace "?" or empty values with NaN (some CKD datasets have "?")
df.replace("?", np.nan, inplace=True)

# 3. Convert all numeric possible columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# 4. Handle missing values properly
imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 5. Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# 6. Split data
X = df.drop("classification", axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
results["Logistic Regression"] = accuracy_score(y_test, lr.predict(X_test))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
results["SVM"] = accuracy_score(y_test, svm.predict(X_test))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
results["Decision Tree"] = accuracy_score(y_test, dt.predict(X_test))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test))

# XGBoost
xg = xgb.XGBClassifier(eval_metric="logloss")
xg.fit(X_train, y_train)
results["XGBoost"] = accuracy_score(y_test, xg.predict(X_test))

print("\nMODEL ACCURACY COMPARISON:")
for model, acc in results.items():
    print(f"{model}: {acc*100:.2f}%")
