"""
Trains a Random Forest that predicts failure risk from sensor readings,
then converts that probability into a 0-100 health score.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

os.makedirs("models", exist_ok=True)

FEATURES = ["temperature", "vibration", "pressure", "rpm", "current"]
TARGET = "failure_risk"

# 1. Load data
df = pd.read_csv("data/sensor_data.csv")

# 2. Split BY MACHINE so the model is tested on machines it has never seen.
#    (Random row splitting would leak information and inflate accuracy.)
train_df = df[df["machine_id"] <= 16]
test_df = df[df["machine_id"] > 16]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

# 3. Train
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    class_weight="balanced",  # handles the 80/20 imbalance
    random_state=42,
)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
}

print("\n=== Evaluation on unseen machines ===")
print(metrics)
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=["Healthy", "At risk"]))

# 5. Which sensors matter most?
importance = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
    .round(3)
)
print("Feature importance:")
print(importance)


# 6. Health score = (1 - failure probability) * 100
def health_score(readings: pd.DataFrame) -> pd.Series:
    failure_prob = model.predict_proba(readings[FEATURES])[:, 1]
    return ((1 - failure_prob) * 100).round(1)


test_df = test_df.copy()
test_df["health_score"] = health_score(test_df)
print("\nAverage health score, healthy readings:",
      round(test_df[test_df[TARGET] == 0]["health_score"].mean(), 1))
print("Average health score, at-risk readings:",
      round(test_df[test_df[TARGET] == 1]["health_score"].mean(), 1))

# 7. Save everything the API will need
joblib.dump({"model": model, "features": FEATURES}, "models/health_model.joblib")
with open("models/metrics.json", "w") as f:
    json.dump({"metrics": metrics, "feature_importance": importance.to_dict()}, f, indent=2)

print("\nSaved models/health_model.joblib and models/metrics.json")