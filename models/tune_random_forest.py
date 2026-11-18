
from pathlib import Path

import pandas as pd
import joblib
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "features" / "model_ready" / "train_labeled.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nPATHSNAP - BAYESIAN HYPERPARAMETER OPTIMIZATION")

df = pd.read_csv(DATA_PATH)

positive_counts = (
    df.groupby(["trip_id", "observation_id"])["label"].transform("sum")
)

df = df[positive_counts == 1].copy()

print("Dataset shape:", df.shape)
print("Unique trips:", df["trip_id"].nunique())


# --------------------------------------------------
# 3. FEATURES AND TARGET
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

X = df[features]
y = df["label"]
groups = df["trip_id"]


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 5. REDUCE DATA FOR OPTIMIZATION
# --------------------------------------------------

# Use half of the training trips for tuning.
# The held-out test data is never used during optimization.

unique_trips = groups_train.unique()

import numpy as np

rng = np.random.RandomState(42)

selected_trips = rng.choice(
    unique_trips,
    size=max(3, len(unique_trips) // 2),
    replace=False
)

tune_mask = groups_train.isin(selected_trips)

X_tune = X_train.loc[tune_mask]
y_tune = y_train.loc[tune_mask]
groups_tune = groups_train.loc[tune_mask]

print("Tuning rows:", len(X_tune))
print("Tuning trips:", len(selected_trips))


# --------------------------------------------------
# 6. DEFINE OBJECTIVE FUNCTION
# --------------------------------------------------

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 200, step=50
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 10, 25, step=5
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 10
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 5
        ),

        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        )
    }

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("classifier", RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    cv = GroupKFold(n_splits=3)

    scores = cross_val_score(
        model,
        X_tune,
        y_tune,
        groups=groups_tune,
        cv=cv,
        scoring="f1",
        n_jobs=1
    )

    return scores.mean()


# --------------------------------------------------
# 7. BAYESIAN OPTIMIZATION USING OPTUNA
# --------------------------------------------------

sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler
)

print("\nStarting Bayesian Optimization...")

study.optimize(
    objective,
    n_trials=10,
    show_progress_bar=True
)


# --------------------------------------------------
# 8. DISPLAY BEST PARAMETERS
# --------------------------------------------------

print("\nBEST HYPERPARAMETERS")

for key, value in study.best_params.items():
    print(key, ":", value)

print("\nBest cross-validation F1:", study.best_value)


# --------------------------------------------------
# 9. TRAIN BASELINE RANDOM FOREST
# --------------------------------------------------

baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining original Random Forest...")

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)


# --------------------------------------------------
# 10. TRAIN OPTIMIZED RANDOM FOREST
# --------------------------------------------------

print("\nTraining optimized Random Forest...")

tuned_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("classifier", RandomForestClassifier(
        **study.best_params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

tuned_model.fit(X_train, y_train)

tuned_predictions = tuned_model.predict(X_test)


# --------------------------------------------------
# 11. EVALUATION
# --------------------------------------------------

def evaluate(y_true, predictions):

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, predictions, zero_division=0
        )
    }


results = pd.DataFrame([
    {
        "Model": "Original Random Forest",
        **evaluate(y_test, baseline_predictions)
    },
    {
        "Model": "Bayesian Optimized Random Forest",
        **evaluate(y_test, tuned_predictions)
    }
])

print("\nFINAL COMPARISON")
print(results.to_string(index=False))


# --------------------------------------------------
# 12. SAVE OUTPUTS
# --------------------------------------------------

joblib.dump(
    tuned_model,
    OUTPUT_DIR / "random_forest_bayesian_tuned.joblib"
)

results.to_csv(
    OUTPUT_DIR / "bayesian_tuning_comparison.csv",
    index=False
)

study.trials_dataframe().to_csv(
    OUTPUT_DIR / "bayesian_optimization_trials.csv",
    index=False
)

joblib.dump(
    study,
    OUTPUT_DIR / "bayesian_optimization_study.joblib"
)

print("\nAll Bayesian Optimization results saved to:")
print(OUTPUT_DIR)
