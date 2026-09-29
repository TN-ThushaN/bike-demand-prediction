import pandas as pd
from pathlib import Path

print("="*60)
print("DATA PREPROCESSING")
print("="*60)

PROJECT_DIR = Path(__file__).resolve().parent.parent

input_file = PROJECT_DIR / "results" / "01_loaded_data.csv"

output_file = PROJECT_DIR / "results" / "02_preprocessed_data.csv"

df = pd.read_csv(input_file)

print("\nDataset loaded.")

df["datetime"] = pd.to_datetime(df["datetime"])

df["rain_1h"] = df["rain_1h"].fillna(0)

df["snow_1h"] = df["snow_1h"].fillna(0)

df["year"] = df["datetime"].dt.year

df["month"] = df["datetime"].dt.month

df["day"] = df["datetime"].dt.day

df["hour"] = df["datetime"].dt.hour

df["weekday"] = df["datetime"].dt.weekday

max_count = df["count"].max()

df["demand_percentage"] = (
    df["count"] / max_count
) * 100

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nMissing values:")

print(df.isnull().sum())

print("\nNew columns:")

print([
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "demand_percentage"
])

print("\nFirst 5 rows:")

print(df.head())

df.to_csv(output_file, index=False)

print("\nSaved:")

print(output_file)

print("\nPREPROCESSING COMPLETED")