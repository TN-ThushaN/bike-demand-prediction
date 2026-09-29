import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline


print("=" * 60)
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 60)


df = pd.read_csv(
    "results/02_preprocessed_data.csv"
)

print("\nDataset loaded successfully.")

print("Dataset shape:", df.shape)


df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


print(
    "\nDate range:",
    df["datetime"].min(),
    "to",
    df["datetime"].max()
)
target = "demand_percentage"

y = df[target]

features = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "holiday",
    "workingday",
    "temp",
    "feels_like",
    "humidity",
    "weather_main"
]

X = df[features]


print("\nFeatures used for prediction:")

for feature in features:
    print("-", feature)


numerical_features = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "holiday",
    "workingday",
    "temp",
    "feels_like",
    "humidity"
]

categorical_features = [
    "weather_main"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


print(
    "\nTraining period:",
    df["datetime"].iloc[0],
    "to",
    df["datetime"].iloc[split_index - 1]
)

print(
    "Testing period:",
    df["datetime"].iloc[split_index],
    "to",
    df["datetime"].iloc[-1]
)

models = {

    "linear_regression":
        LinearRegression(),

    "decision_tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "random_forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "gradient_boosting":
        GradientBoostingRegressor(
            random_state=42
        )
}

trained_models = {}


for name, model in models.items():

    print("\n" + "-" * 60)

    print(
        "Training:",
        name
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    trained_models[name] = pipeline

    print(
        "Training completed:",
        name
    )


joblib.dump(
    trained_models["linear_regression"],
    "models/linear_regression.joblib"
)

joblib.dump(
    trained_models["decision_tree"],
    "models/decision_tree.joblib"
)

joblib.dump(
    trained_models["random_forest"],
    "models/random_forest.joblib"
)

joblib.dump(
    trained_models["gradient_boosting"],
    "models/gradient_boosting.joblib"
)

X_test.to_csv(
    "results/X_test.csv",
    index=False
)

y_test.to_csv(
    "results/y_test.csv",
    index=False
)

joblib.dump(
    features,
    "models/feature_columns.joblib"
)

joblib.dump(
    target,
    "models/target_column.joblib"
)

print("\n")
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nSaved models:")

print("models/linear_regression.joblib")
print("models/decision_tree.joblib")
print("models/random_forest.joblib")
print("models/gradient_boosting.joblib")

print("\nSaved files:")

print("results/X_test.csv")
print("results/y_test.csv")
print("models/feature_columns.joblib")
print("models/target_column.joblib")

print("\nFeatures now used for future prediction:")

for feature in features:
    print("-", feature)

print("\nNext step:")
print("Run 05_model_evaluation.py")