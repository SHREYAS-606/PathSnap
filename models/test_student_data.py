
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    ROOT / "data" / "features" /
    "model_ready" / "student_test_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "model_results"

PREDICTION_PATH = OUTPUT_DIR / "student_test_predictions.csv"

# Find the existing best model, including inside saved models.
possible_models = list(OUTPUT_DIR.rglob("best_model.joblib"))

if not possible_models:
    raise FileNotFoundError(
        "best_model.joblib not found inside outputs/model_results. "
        "Check your saved models folder."
    )

MODEL_PATH = possible_models[0]

print("\nPATHSNAP - STUDENT TEST EVALUATION")
print("Model:", MODEL_PATH)
print("Test data:", TEST_PATH)


# --------------------------------------------------
# 2. LOAD MODEL AND DATA
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
df = pd.read_csv(TEST_PATH)

print("\nStudent test dataset shape:", df.shape)


# --------------------------------------------------
# 3. SELECT THE SAME FEATURES USED IN TRAINING
# --------------------------------------------------

features = [
    "distance_m",
    "projection_fraction",
    "segment_length_m",
    "longitude",
    "latitude",
    "timestamp",
    "time_gap_s",
    "gps_speed_mps",
    "gps_heading_deg",
    "road_heading_deg",
    "road_length_m",
    "from_node_out_degree",
    "to_node_in_degree",
    "heading_difference_deg",
    "distance_squared",
    "is_valid_candidate",
    "large_time_gap"
]

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

X_test = df[features]


# --------------------------------------------------
# 4. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating student test predictions...")

predictions = model.predict(X_test)

df["predicted_label"] = predictions

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test)

    positive_class_index = list(
        model.classes_
    ).index(1)

    df["positive_probability"] = (
        probabilities[:, positive_class_index]
    )


# --------------------------------------------------
# 5. EVALUATE IF GROUND-TRUTH LABELS EXIST
# --------------------------------------------------

if "label" in df.columns:

    print("\nSTUDENT TEST EVALUATION")

    print("Accuracy:",
          accuracy_score(df["label"], predictions))

    print("Precision:",
          precision_score(
              df["label"], predictions, zero_division=0
          ))

    print("Recall:",
          recall_score(
              df["label"], predictions, zero_division=0
          ))

    print("F1 Score:",
          f1_score(
              df["label"], predictions, zero_division=0
          ))

else:
    print("\nNo ground-truth label column found.")
    print("Predictions generated; evaluation metrics unavailable.")


# --------------------------------------------------
# 6. SAVE PREDICTIONS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(PREDICTION_PATH, index=False)

print("\nPrediction counts:")
print(df["predicted_label"].value_counts())

print("\nStudent test predictions saved to:")
print(PREDICTION_PATH)
