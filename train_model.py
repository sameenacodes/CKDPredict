import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier


data = pd.read_csv("Kidney_data.csv")
data = data.replace("?", pd.NA)

# Selected features used in web app
features = ["sg", "al", "hemo", "rc", "htn", "dm", "appet", "pc"]
X = data[features]
y = data["classification"]

# Encode categorical columns
for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Encode target column
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Fill missing values
X = X.fillna(X.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump({"model": model, "accuracy": accuracy}, open("Kidney.pkl", "wb"))

# ===== Feature Importance Graph =====
plt.figure(figsize=(8,5))
plt.bar(features, model.feature_importances_)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/feature_importance.png")

# ===== Confusion Matrix =====
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion_matrix.png")

print("Training complete. Graphs saved!")


