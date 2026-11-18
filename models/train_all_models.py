
from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "features" / "model_ready" / "train_labeled.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_results"

TARGET = "label"

DROP_COLUMNS = [
    "trip_id", "partition", "observation_id",
    "candidate_arc_id", "from_node", "to_node",
    "label", "label_arc_id", "label_route_position",
    "label_route_distance_m", "label_available"
]


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError("The dataset must contain a 'label' column.")

    # Keep only observations with exactly one positive candidate.
    group_cols = ["trip_id", "observation_id"]
    positive_counts = df.groupby(group_cols)[TARGET].transform("sum")
    df = df[positive_counts == 1].copy()

    print("Dataset loaded:", df.shape)
    print("Unique trips:", df["trip_id"].nunique())
    print("Positive labels:", int(df[TARGET].sum()))

    return df


def main():
    print("\nPATHSNAP - FIVE MODEL COMPARISON\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in DROP_COLUMNS
    ]

    X = df[feature_cols]
    y = df[TARGET].astype(int)
    groups = df["trip_id"]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Features:", feature_cols)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=12, class_weight="balanced", random_state=42
            ))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=15,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=3, random_state=42
            ))
        ])
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        elapsed = time.time() - start

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Training Time (seconds)": round(elapsed, 2)
        }

        results.append(metrics)

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, OUTPUT_DIR / filename)

        print(metrics)

    comparison = pd.DataFrame(results).sort_values(
        "F1 Score", ascending=False
    )

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_name = comparison.iloc[0]["Model"]
    best_file = best_name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(
        joblib.load(OUTPUT_DIR / best_file),
        OUTPUT_DIR / "best_model.joblib"
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))
    print("\nBest model:", best_name)
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
