import pandas as pd
from pathlib import Path



PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "capitalbikeshare-complete.csv"

OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found.\n"
        f"Please put 'capitalbikeshare-complete.csv' inside:\n"
        f"{PROJECT_DIR / 'data'}"
    )


df = pd.read_csv(DATA_FILE)



print("\n" + "=" * 60)
print("BIKE-SHARING DATASET - DATA LOADING")
print("=" * 60)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


print("\n" + "-" * 60)
print("COLUMN NAMES")
print("-" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


print("\n" + "-" * 60)
print("FIRST 5 ROWS")
print("-" * 60)

print(df.head())


print("\n" + "-" * 60)
print("DATA TYPES")
print("-" * 60)

print(df.dtypes)


print("\n" + "-" * 60)
print("MISSING VALUES")
print("-" * 60)

missing_values = df.isnull().sum()

print(missing_values)


print("\n" + "-" * 60)
print("DUPLICATE ROWS")
print("-" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


print("\n" + "-" * 60)
print("DATETIME INFORMATION")
print("-" * 60)

if "datetime" in df.columns:

    datetime_values = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    print("First datetime:", datetime_values.min())
    print("Last datetime:", datetime_values.max())

    print(
        "Invalid datetime values:",
        datetime_values.isna().sum()
    )

else:
    print("WARNING: 'datetime' column was not found.")


print("\n" + "-" * 60)
print("NUMERICAL SUMMARY")
print("-" * 60)

print(df.describe())


output_file = OUTPUT_DIR / "01_loaded_data.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("DATA LOADING COMPLETED")
print("=" * 60)

print("\nLoaded data saved to:")
print(output_file)

print("\nNext step:")
print("Run 02_data_preprocessing.py")