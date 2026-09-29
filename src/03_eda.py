import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "results" / "02_preprocessed_data.csv"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])


print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)



print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(df["datetime"].min())
print("to")
print(df["datetime"].max())



print("\n" + "-" * 60)
print("DEMAND STATISTICS")
print("-" * 60)

print(df["count"].describe())


print("\n" + "-" * 60)
print("DEMAND LEVEL (%) STATISTICS")
print("-" * 60)

print(df["demand_percentage"].describe())


hourly_demand = (
    df.groupby("hour")["demand_percentage"]
    .mean()
    .reset_index()
)

print("\n" + "-" * 60)
print("AVERAGE DEMAND BY HOUR")
print("-" * 60)

print(hourly_demand)



hourly_demand.to_csv(
    RESULTS_DIR / "average_demand_by_hour.csv",
    index=False
)


monthly_demand = (
    df.groupby("month")["demand_percentage"]
    .mean()
    .reset_index()
)

print("\n" + "-" * 60)
print("AVERAGE DEMAND BY MONTH")
print("-" * 60)

print(monthly_demand)


monthly_demand.to_csv(
    RESULTS_DIR / "average_demand_by_month.csv",
    index=False
)


yearly_demand = (
    df.groupby("year")["demand_percentage"]
    .mean()
    .reset_index()
)

print("\n" + "-" * 60)
print("AVERAGE DEMAND BY YEAR")
print("-" * 60)

print(yearly_demand)


yearly_demand.to_csv(
    RESULTS_DIR / "average_demand_by_year.csv",
    index=False
)


workingday_demand = (
    df.groupby("workingday")["demand_percentage"]
    .mean()
    .reset_index()
)

print("\n" + "-" * 60)
print("WORKING DAY VS NON-WORKING DAY")
print("-" * 60)

print(workingday_demand)


workingday_demand.to_csv(
    RESULTS_DIR / "demand_workingday.csv",
    index=False
)


weather_demand = (
    df.groupby("weather_main")["demand_percentage"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print("\n" + "-" * 60)
print("AVERAGE DEMAND BY WEATHER")
print("-" * 60)

print(weather_demand)


weather_demand.to_csv(
    RESULTS_DIR / "average_demand_by_weather.csv",
    index=False
)


print("\n" + "-" * 60)
print("CORRELATION WITH DEMAND")
print("-" * 60)

temp_correlation = df["temp"].corr(
    df["demand_percentage"]
)

humidity_correlation = df["humidity"].corr(
    df["demand_percentage"]
)

print("Temperature correlation:", temp_correlation)
print("Humidity correlation:", humidity_correlation)


plt.figure(figsize=(10, 5))

plt.plot(
    hourly_demand["hour"],
    hourly_demand["demand_percentage"],
    marker="o"
)

plt.title("Average Bike Demand by Hour")

plt.xlabel("Hour")

plt.ylabel("Demand Level (%)")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "demand_by_hour.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_demand["month"],
    monthly_demand["demand_percentage"],
    marker="o"
)

plt.title("Average Bike Demand by Month")

plt.xlabel("Month")

plt.ylabel("Demand Level (%)")

plt.xticks(range(1, 13))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "demand_by_month.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

plt.bar(
    yearly_demand["year"].astype(str),
    yearly_demand["demand_percentage"]
)

plt.title("Average Bike Demand by Year")

plt.xlabel("Year")

plt.ylabel("Demand Level (%)")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "demand_by_year.png",
    dpi=300
)

plt.show()


workingday_labels = {
    0: "Non-working day",
    1: "Working day"
}

workingday_plot = workingday_demand.copy()

workingday_plot["day_type"] = (
    workingday_plot["workingday"]
    .map(workingday_labels)
)

plt.figure(figsize=(8, 5))

plt.bar(
    workingday_plot["day_type"],
    workingday_plot["demand_percentage"]
)

plt.title("Average Demand: Working vs Non-working Days")

plt.xlabel("Day Type")

plt.ylabel("Demand Level (%)")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "demand_workingday.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(12, 6))

plt.bar(
    weather_demand["weather_main"],
    weather_demand["demand_percentage"]
)

plt.title("Average Bike Demand by Weather Condition")

plt.xlabel("Weather Condition")

plt.ylabel("Demand Level (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "demand_by_weather.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    df["temp"],
    df["demand_percentage"],
    alpha=0.3
)

plt.title("Temperature vs Bike Demand")

plt.xlabel("Temperature")

plt.ylabel("Demand Level (%)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "temperature_vs_demand.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    df["humidity"],
    df["demand_percentage"],
    alpha=0.3
)

plt.title("Humidity vs Bike Demand")

plt.xlabel("Humidity (%)")

plt.ylabel("Demand Level (%)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "humidity_vs_demand.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nEDA results saved inside:")
print(RESULTS_DIR)

print("\nNext step:")
print("Run 04_model_training.py")