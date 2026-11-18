
from pathlib import Path
import pandas as pd
import joblib

# Project directories
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "features" / "engineered" / "student_test"
MODEL_PATH = ROOT / "outputs" / "model_results" / "best_model.joblib"
OUTPUT_DIR = ROOT / "outputs" / "model_results" / "student_test_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same 17 features used during model training
FEATURES = [
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


def main():

    print("\nPATHSNAP - ENGINEERED STUDENT TESTING")
    print("=" * 50)

    # Load trained model
    print("\nLoading trained Random Forest model...")
    model = joblib.load(MODEL_PATH)

    # Find all student test files
    files = sorted(INPUT_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No student test CSV files found!")

    print("Total student test trips:", len(files))

    all_results = []

    # Process each trip
    for file in files:

        print(f"\nTesting trip: {file.name}")

        df = pd.read_csv(file)

        missing = [col for col in FEATURES if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing features in {file.name}: {missing}"
            )

        X = df[FEATURES]

        # Generate predictions
        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        # Preserve original data and add predictions
        df["predicted_label"] = predictions
        df["positive_probability"] = probabilities

        df["source_file"] = file.name

        # Save individual trip result
        individual_output = OUTPUT_DIR / (
            file.stem + "_predictions.csv"
        )

        df.to_csv(individual_output, index=False)

        all_results.append(df)

        print("Candidate rows:", len(df))
        print("Predicted positive:", int((predictions == 1).sum()))
        print("Predicted negative:", int((predictions == 0).sum()))

    # Combine all trips
    combined = pd.concat(all_results, ignore_index=True)

    combined_output = OUTPUT_DIR / "all_student_test_predictions.csv"

    combined.to_csv(combined_output, index=False)

    # Summary for all trips
    summary = (
        combined.groupby("source_file")
        .agg(
            total_candidates=("predicted_label", "size"),
            positive_predictions=("predicted_label", "sum"),
            average_probability=("positive_probability", "mean")
        )
        .reset_index()
    )

    summary_output = OUTPUT_DIR / "student_test_summary.csv"
    summary.to_csv(summary_output, index=False)

    print("\n" + "=" * 50)
    print("STUDENT TESTING COMPLETED")
    print("=" * 50)

    print("\nTotal candidate rows:", len(combined))
    print("Total positive predictions:",
          int((combined["predicted_label"] == 1).sum()))

    print("\nTrip-wise summary:")
    print(summary.to_string(index=False))

    print("\nResults saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
