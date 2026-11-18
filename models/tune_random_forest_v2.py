
from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve()

for parent in ROOT.parents:
    candidate = parent / "data" / "features" / "model_ready" / "train_labeled.csv"
    if candidate.exists():
        ROOT = parent
        break
else:
    raise FileNotFoundError("PathSnap training dataset not found.")

DATA_PATH = ROOT / "data" / "features" / "model_ready" / "train_labeled.csv"

OUTPUT_DIR = ROOT / "outputs" / "model_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS = 20
OBSERVATIONS_PER_TRIP = 500

print("\nPATHSNAP - RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

positive_counts = (
    df.groupby(["trip_id", "observation_id"])["label"].transform("sum")
)

df = df[positive_counts == 1].copy()

print("Dataset shape:", df.shape)
print("Unique trips:", df["trip_id"].nunique())

# ==================================================
# 3. FEATURES AND TARGET
# SAME FEATURE SELECTION AS TRAIN_ALL_MODELS
# ==================================================

DROP_COLUMNS = [
    "trip_id",
    "partition",
    "observation_id",
    "candidate_arc_id",
    "from_node",
    "to_node",
    "label",
    "label_arc_id",
    "label_route_position",
    "label_route_distance_m",
    "label_available"
]

features = [
    column
    for column in df.select_dtypes(include="number").columns
    if column not in DROP_COLUMNS
]

X = df[features]
y = df["label"].astype(int)
groups = df["trip_id"]

print("\nNumber of features:", len(features))
print("Features:", features)

# ==================================================
# 4. GROUP-BASED TRAIN TEST SPLIT
# ==================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training trips:", groups_train.nunique())
print("Testing trips:", groups.iloc[test_idx].nunique())

# ==================================================
# 5. CREATE TUNING SUBSET
# TEST DATA IS NEVER USED FOR TUNING
# ==================================================

rng = np.random.RandomState(RANDOM_STATE)

unique_trips = groups_train.unique()

selected_trips = rng.choice(
    unique_trips,
    size=max(3, len(unique_trips) // 2),
    replace=False
)

tune_mask = groups_train.isin(selected_trips)

X_tune = X_train.loc[tune_mask]
y_tune = y_train.loc[tune_mask]
groups_tune = groups_train.loc[tune_mask]

# Sample observations, retaining every candidate
# belonging to each selected observation.

tune_frame = df.loc[X_tune.index]

selected_observations = []

for trip_id, part in tune_frame.groupby("trip_id"):

    observations = part["observation_id"].drop_duplicates().to_numpy()

    chosen = rng.choice(
        observations,
        size=min(OBSERVATIONS_PER_TRIP, len(observations)),
        replace=False
    )

    selected_observations.extend(
        (trip_id, observation)
        for observation in chosen
    )

selected_set = set(selected_observations)

keep = [
    (trip, observation) in selected_set
    for trip, observation in zip(
        tune_frame["trip_id"],
        tune_frame["observation_id"]
    )
]

selected_indices = tune_frame.index[np.array(keep)]

X_tune = X_tune.loc[selected_indices]
y_tune = y_tune.loc[selected_indices]
groups_tune = groups_tune.loc[selected_indices]

print("\nTuning rows:", len(X_tune))
print("Tuning trips:", groups_tune.nunique())

if groups_tune.nunique() < 3:
    raise ValueError("At least three training trips are required.")

# ==================================================
# 6. RANDOM FOREST PIPELINE
# ==================================================

def create_model(params):

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                **params,
                bootstrap=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])

# ==================================================
# 7. BAYESIAN OPTIMIZATION OBJECTIVE
# ==================================================

def objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 400, step=50
        ),

        "max_depth": trial.suggest_categorical(
            "max_depth", [10, 15, 20, 25, 30, None]
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 20
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 10
        ),

        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.7]
        ),

        "max_samples": trial.suggest_categorical(
            "max_samples", [0.7, 0.85, 1.0]
        ),

        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy"]
        )
    }

    model = create_model(params)

    cv = GroupKFold(n_splits=3)

    scores = cross_val_score(
        model,
        X_tune,
        y_tune,
        groups=groups_tune,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    trial.set_user_attr("fold_scores", scores.tolist())

    return scores.mean()

# ==================================================
# 8. RUN BAYESIAN OPTIMIZATION
# ==================================================

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    ),
    study_name="PathSnap_Random_Forest_Optimization"
)

print("\nSTARTING BAYESIAN OPTIMIZATION")
print("Optimization trials:", N_TRIALS)
print("Cross-validation folds: 3")
print("Optimization metric: Accuracy")

start_time = time.time()

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)

optimization_time = time.time() - start_time

# ==================================================
# 9. DISPLAY BEST PARAMETERS
# ==================================================

print("\nBEST RANDOM FOREST PARAMETERS")
print("=" * 50)

for parameter, value in study.best_params.items():
    print(f"{parameter}: {value}")

print("\nBest CV Accuracy:", study.best_value)

# ==================================================
# 10. TRAIN BASELINE RANDOM FOREST
# ==================================================

baseline_params = {
    "n_estimators": 100,
    "max_depth": 15
}

baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        **baseline_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

print("\nTRAINING BASELINE RANDOM FOREST...")

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

# ==================================================
# 11. TRAIN TUNED RANDOM FOREST
# ==================================================

print("\nTRAINING BAYESIAN TUNED RANDOM FOREST...")

tuned_model = create_model(study.best_params)

tuned_model.fit(X_train, y_train)

tuned_predictions = tuned_model.predict(X_test)

# ==================================================
# 12. EVALUATION
# ==================================================

def evaluate_model(y_true, predictions):

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

baseline_metrics = evaluate_model(
    y_test, baseline_predictions
)

tuned_metrics = evaluate_model(
    y_test, tuned_predictions
)

results = pd.DataFrame([
    {
        "Model": "Baseline Random Forest",
        **baseline_metrics
    },
    {
        "Model": "Bayesian Tuned Random Forest",
        **tuned_metrics
    }
])

# ==================================================
# 13. DISPLAY COMPARISON
# ==================================================

print("\n" + "=" * 65)
print("BASELINE VS TUNED RANDOM FOREST")
print("=" * 65)

print(results.to_string(index=False))

accuracy_change = (
    tuned_metrics["Accuracy"] - baseline_metrics["Accuracy"]
) * 100

print("\nAccuracy improvement:", round(accuracy_change, 4), "percentage points")

if accuracy_change > 0:
    print("Tuned Random Forest improved test accuracy!")
elif accuracy_change < 0:
    print("Baseline Random Forest has better test accuracy.")
else:
    print("Both models have equal test accuracy.")

print("\nOptimization time:", round(optimization_time, 2), "seconds")

# ==================================================
# 14. SAVE MODELS AND RESULTS
# ==================================================

joblib.dump(
    baseline_model,
    OUTPUT_DIR / "random_forest_baseline.joblib"
)

joblib.dump(
    tuned_model,
    OUTPUT_DIR / "random_forest_bayesian_tuned.joblib"
)

# Save the better test-accuracy model separately.
best_model = (
    tuned_model
    if tuned_metrics["Accuracy"] > baseline_metrics["Accuracy"]
    else baseline_model
)

joblib.dump(
    best_model,
    OUTPUT_DIR / "random_forest_best.joblib"
)

results.to_csv(
    OUTPUT_DIR / "random_forest_tuning_comparison.csv",
    index=False
)

study.trials_dataframe().to_csv(
    OUTPUT_DIR / "random_forest_bayesian_trials.csv",
    index=False
)

joblib.dump(
    study,
    OUTPUT_DIR / "random_forest_bayesian_study.joblib"
)

print("\nALL RESULTS SAVED SUCCESSFULLY!")
print("Output directory:", OUTPUT_DIR)

print("\nTUNING COMPLETED!")