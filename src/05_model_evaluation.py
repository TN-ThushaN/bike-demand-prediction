import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


X_test = pd.read_csv(
    "results/X_test.csv"
)

y_test = pd.read_csv(
    "results/y_test.csv"
).squeeze()


print("\nTest data loaded.")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


models = {

    "Linear Regression":
        joblib.load(
            "models/linear_regression.joblib"
        ),

    "Decision Tree":
        joblib.load(
            "models/decision_tree.joblib"
        ),

    "Random Forest":
        joblib.load(
            "models/random_forest.joblib"
        ),

    "Gradient Boosting":
        joblib.load(
            "models/gradient_boosting.joblib"
        )
}

results = []

predictions = {}


for name, model in models.items():

    print("\nEvaluating:", name)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

prediction_df = pd.DataFrame({
    "Actual": y_test.values,

    "Linear_Regression":
        predictions["Linear Regression"],

    "Decision_Tree":
        predictions["Decision Tree"],

    "Random_Forest":
        predictions["Random Forest"],

    "Gradient_Boosting":
        predictions["Gradient Boosting"]
})


prediction_df.to_csv(
    "results/demand_predictions.csv",
    index=False
)

gb_predictions = predictions[
    "Gradient Boosting"
]

errors = (
    gb_predictions - y_test.values
)

absolute_errors = np.abs(
    errors
)

mean_error = np.mean(
    errors
)

mean_absolute_error_value = np.mean(
    absolute_errors
)

maximum_absolute_error = np.max(
    absolute_errors
)


print("\n")
print("=" * 60)
print("GRADIENT BOOSTING ERROR ANALYSIS")
print("=" * 60)

print(
    f"Mean Error: "
    f"{mean_error:.6f}"
)

print(
    f"Mean Absolute Error: "
    f"{mean_absolute_error_value:.6f}"
)

print(
    f"Maximum Absolute Error: "
    f"{maximum_absolute_error:.6f}"
)

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    y_test.values[:300],
    label="Actual"
)

plt.plot(
    gb_predictions[:300],
    label="Gradient Boosting"
)

plt.title(
    "Actual vs Predicted Bike Demand"
)

plt.xlabel(
    "Test Observation"
)

plt.ylabel(
    "Demand Level (%)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300
)

plt.close()

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title(
    "Model Comparison - R²"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R²"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "results/model_comparison_r2.png",
    dpi=300
)

plt.close()

print("\n")
print("=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print("\nSaved files:")

print("results/model_comparison.csv")
print("results/demand_predictions.csv")
print("results/actual_vs_predicted.png")
print("results/model_comparison_r2.png")

print("\nNext step:")
print("Run 06_feature_importance.py")