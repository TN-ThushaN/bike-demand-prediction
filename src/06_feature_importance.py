import pandas as pd
import joblib
import matplotlib.pyplot as plt


print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

model = joblib.load(
    "models/gradient_boosting.joblib"
)

print("\nGradient Boosting model loaded.")

features = joblib.load(
    "models/feature_columns.joblib"
)

trained_model = model.named_steps["model"]


preprocessor = model.named_steps[
    "preprocessor"
]

feature_names = (
    preprocessor
    .get_feature_names_out()
)

feature_names = [
    name.split("__")[-1]
    for name in feature_names
]

importance = (
    trained_model.feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nFeature importance:")

print(
    importance_df.to_string(
        index=False
    )
)

importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "Gradient Boosting Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)

plt.close()

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 60)

print("\nSaved files:")

print("results/feature_importance.csv")
print("results/feature_importance.png")

print("\nNext step:")
print("Update 07_future_prediction.py")