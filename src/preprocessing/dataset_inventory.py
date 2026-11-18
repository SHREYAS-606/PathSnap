
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

FILE_TYPES = ["track", "nodes", "arcs", "route"]
PARTITIONS = ["train", "student_test"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

for partition in PARTITIONS:
    partition_dir = DATA_DIR / partition

    if not partition_dir.exists():
        print(f"Missing partition: {partition}")
        continue

    trips = sorted(
        p for p in partition_dir.iterdir() if p.is_dir()
    )

    print(f"\nPartition: {partition}")
    print(f"Number of trips: {len(trips)}")

    for trip in trips:
        for extension in FILE_TYPES:
            file_path = trip / f"{trip.name}.{extension}"

            exists = file_path.exists()
            row_count = 0
            columns = None
            size = 0

            if exists:
                size = file_path.stat().st_size

                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    columns = len(first_line.split()) if first_line else 0
                    row_count = 1 if first_line else 0

                    for line in f:
                        if line.strip():
                            row_count += 1

            records.append({
                "partition": partition,
                "trip_id": trip.name,
                "file_type": extension,
                "exists": exists,
                "rows": row_count,
                "columns": columns,
                "size_bytes": size
            })

df = pd.DataFrame(records)

output_file = OUTPUT_DIR / "dataset_inventory.csv"
df.to_csv(output_file, index=False)

print("\n========== DATASET SUMMARY ==========")

print(
    df.groupby(["partition", "file_type"])
    .agg(
        files=("exists", "sum"),
        total_rows=("rows", "sum"),
        total_size_bytes=("size_bytes", "sum")
    )
)

print("\nMissing files:")
print(df[~df["exists"]])

print(f"\nInventory saved to: {output_file}")
