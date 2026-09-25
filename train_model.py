import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier


# ============================================================
# CKDPredict - MODEL TRAINING
# ============================================================

print("=" * 60)
print("CKDPredict - MODEL TRAINING")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("Kidney_data.csv")

print("\nOriginal target values:")
print(data["classification"].value_counts(dropna=False))


# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Replace ? with missing values
data = data.replace("?", pd.NA)

# Remove unwanted spaces / tabs from string values
for col in data.columns:
    if data[col].dtype == object:
        data[col] = (
            data[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )


# ============================================================
# 3. CLEAN TARGET COLUMN
# ============================================================

# IMPORTANT:
# "ckd\t" becomes "ckd"
data["classification"] = (
    data["classification"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nCleaned target values:")
print(data["classification"].value_counts(dropna=False))


# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    "sg",
    "al",
    "hemo",
    "rc",
    "htn",
    "dm",
    "appet",
    "pc"
]

X = data[features].copy()
y = data["classification"].copy()


# ============================================================
# 5. CONVERT NUMERIC FEATURES
# ============================================================

numeric_features = [
    "sg",
    "al",
    "hemo",
    "rc"
]

for col in numeric_features:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# ============================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================

# ------------------------------------------------------------
# HTN
# no  = 0
# yes = 1
# ------------------------------------------------------------

X["htn"] = X["htn"].map({
    "no": 0,
    "yes": 1
})


# ------------------------------------------------------------
# DM
# no  = 0
# yes = 1
# ------------------------------------------------------------

X["dm"] = X["dm"].map({
    "no": 0,
    "yes": 1
})


# ------------------------------------------------------------
# APPETITE
# good = 0
# poor = 1
# ------------------------------------------------------------

X["appet"] = X["appet"].map({
    "good": 0,
    "poor": 1
})


# ------------------------------------------------------------
# PUS CELL
# abnormal = 0
# normal   = 1
# ------------------------------------------------------------

X["pc"] = X["pc"].map({
    "abnormal": 0,
    "normal": 1
})


# ============================================================
# 7. REMOVE INVALID TARGET VALUES
# ============================================================

valid_targets = [
    "ckd",
    "notckd"
]

valid_rows = y.isin(valid_targets)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()


# ============================================================
# 8. ENCODE TARGET
# ============================================================

# IMPORTANT:
#
# CKD     = 0
# No CKD  = 1
#
# Explicit mapping is used instead of LabelEncoder
# so that Flask and training use the same mapping.

y = y.map({
    "ckd": 0,
    "notckd": 1
})


# ============================================================
# 9. CONVERT ALL FEATURES TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 10. HANDLE MISSING VALUES
# ============================================================

# Fill missing values with column median
X = X.fillna(
    X.median(numeric_only=True)
)

# Safety fallback
X = X.fillna(0)


# ============================================================
# 11. DISPLAY DATASET INFORMATION
# ============================================================

print("\nFeatures used:")
print(features)

print("\nFeature data types:")
print(X.dtypes)

print("\nTarget classes:")
print(sorted(y.unique()))

print("\nTarget distribution:")
print(y.value_counts())

print("\nFinal dataset shape:")
print(X.shape)


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 13. TRAIN XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 14. MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 15. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 16. DISPLAY MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "CKD",
            "No CKD"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

model_data = {
    "model": model,
    "accuracy": accuracy,
    "features": features,

    "class_mapping": {
        0: "ckd",
        1: "notckd"
    }
}


with open(
    "Kidney.pkl",
    "wb"
) as file:

    pickle.dump(
        model_data,
        file
    )


print("\nModel saved successfully:")
print("Kidney.pkl")


# ============================================================
# 18. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    features,
    model.feature_importances_
)

plt.title(
    "CKDPredict - Feature Importance"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "static/feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Feature importance graph saved."
)


# ============================================================
# 19. CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="g",
    xticklabels=[
        "CKD",
        "No CKD"
    ],
    yticklabels=[
        "CKD",
        "No CKD"
    ]
)

plt.title(
    "CKDPredict - Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "static/confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Confusion matrix graph saved."
)


# ============================================================
# 20. FINAL MODEL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL INFORMATION")
print("=" * 60)

print(
    "\nClasses:",
    model.classes_
)

print(
    "Features:",
    model.feature_names_in_
)

print(
    "Number of features:",
    model.n_features_in_
)

print("\nClass mapping:")
print("0 = CKD")
print("1 = No CKD")


# ============================================================
# 21. TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)