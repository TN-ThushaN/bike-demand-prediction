import os
import joblib
import pandas as pd

MODEL_PATH = "models/gradient_boosting.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"

MAX_BIKE_COUNT = 2038

ALLOWED_WEATHER = [
    "Clear",
    "Clouds",
    "Drizzle",
    "Fog",
    "Haze",
    "Mist",
    "Rain",
    "Smoke",
    "Snow",
    "Squall",
    "Thunderstorm"
]

def get_binary_input(message):
    while True:
        value = input(message).strip()

        if value in ["0", "1"]:
            return int(value)

        print("Please enter only 0 or 1.")


def get_float_input(message):
    while True:
        value = input(message).strip()

        try:
            return float(value)

        except ValueError:
            print("Please enter a valid number.")


def get_weather_input():
    while True:
        print("\nAvailable weather conditions:")

        for weather in ALLOWED_WEATHER:
            print(f"- {weather}")

        weather = input("\nWeather condition: ").strip()

        for allowed_weather in ALLOWED_WEATHER:
            if weather.lower() == allowed_weather.lower():
                return allowed_weather

        print("Invalid weather condition. Please choose one from the list.")


print("=" * 60)
print("FUTURE BIKE DEMAND PREDICTION")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Gradient Boosting model was not found.")
    print(f"Expected file: {MODEL_PATH}")
    exit()

if not os.path.exists(FEATURE_COLUMNS_PATH):
    print("\nERROR: Feature columns file was not found.")
    print(f"Expected file: {FEATURE_COLUMNS_PATH}")
    exit()


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("\nGradient Boosting model loaded successfully.")


print("\n" + "-" * 60)
print("ENTER FUTURE CONDITIONS")
print("-" * 60)

while True:

    datetime_input = input(
        "\nFuture date and time "
        "(example: 2026-11-06 17:00): "
    ).strip()

    try:
        future_datetime = pd.to_datetime(datetime_input)

        break

    except Exception:
        print(
            "Invalid date/time format. "
            "Please use YYYY-MM-DD HH:MM."
        )

holiday = get_binary_input(
    "Holiday? (0 = No, 1 = Yes): "
)

workingday = get_binary_input(
    "Working day? (0 = No, 1 = Yes): "
)

temp = get_float_input(
    "Temperature: "
)

feels_like = get_float_input(
    "Feels-like temperature: "
)

humidity = get_float_input(
    "Humidity (%): "
)

weather_main = get_weather_input()


future_data = pd.DataFrame({
    "year": [future_datetime.year],
    "month": [future_datetime.month],
    "day": [future_datetime.day],
    "hour": [future_datetime.hour],
    "weekday": [future_datetime.weekday()],
    "holiday": [holiday],
    "workingday": [workingday],
    "temp": [temp],
    "feels_like": [feels_like],
    "humidity": [humidity],
    "weather_main": [weather_main]
})

predicted_demand_percentage = model.predict(
    future_data[feature_columns]
)[0]

predicted_demand_percentage = max(
    0,
    predicted_demand_percentage
)


estimated_bike_count = (
    predicted_demand_percentage / 100
) * MAX_BIKE_COUNT

estimated_bike_count = round(estimated_bike_count)


print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    f"\nFuture Date/Time: "
    f"{future_datetime.strftime('%Y-%m-%d %H:%M')}"
)

print(
    f"Predicted Demand Level: "
    f"{predicted_demand_percentage:.2f}%"
)

print(
    f"Estimated Bike Count: "
    f"{estimated_bike_count} bikes"
)

print("=" * 60)

print("\nPrediction completed successfully.")

print(
    "\nNOTE:"
    "\nThis is a scenario-based future estimate."
    "\nThe model uses the future date, calendar information,"
    "\nand weather conditions supplied by the user."
    "\nIt does not automatically know future weather conditions."
)

print("=" * 60)