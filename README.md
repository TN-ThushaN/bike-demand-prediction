# 🚲 Bike Demand Prediction System

A machine learning-based application for predicting and analyzing bike-sharing demand using historical rental, calendar, and weather-related data.

The project compares multiple regression algorithms and provides a small Streamlit application that allows users to enter future conditions and generate a scenario-based bike demand prediction.

---

## 📌 Project Overview

Bike-sharing demand changes according to factors such as:

- Time of day
- Month and season
- Working days
- Holidays
- Temperature
- Feels-like temperature
- Humidity
- Weather conditions

The purpose of this project is to analyze these factors and develop machine learning models capable of estimating bike-sharing demand.

The final system includes:

1. Data loading
2. Data preprocessing
3. Exploratory Data Analysis (EDA)
4. Machine learning model training
5. Model evaluation
6. Feature importance analysis
7. Future demand prediction
8. Streamlit prediction application

---

## 🎯 Project Objectives

- Analyze historical bike-sharing demand.
- Identify important factors affecting bike demand.
- Prepare the dataset for machine learning.
- Train and compare multiple regression models.
- Evaluate models using MAE, RMSE, and R².
- Identify the most important prediction factors.
- Develop a small application for future demand estimation.

---

## 🤖 Machine Learning Models

The project compares four regression models:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Gradient Boosting Regression

The models are evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

Based on the current evaluation results, **Gradient Boosting Regression** is selected as the final model.

### Current Evaluation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 5.2987 | 7.4983 | 0.6620 |
| Random Forest | 5.1623 | 7.7113 | 0.6425 |
| Decision Tree | 6.0684 | 9.5472 | 0.4520 |
| Linear Regression | 8.1824 | 11.2231 | 0.2428 |

The final model selection should be discussed together with all three evaluation metrics rather than relying on a single metric.

---

## 📊 Dataset

The project uses a historical bike-sharing dataset containing:

- Date and time
- Bike rental count
- Holiday information
- Working-day information
- Temperature
- Feels-like temperature
- Minimum temperature
- Maximum temperature
- Atmospheric pressure
- Humidity
- Wind speed
- Wind direction
- Rain
- Snow
- Cloud coverage
- Weather condition

The dataset contains:

**33,379 records and 16 original columns.**

The observed date range is:

**2018-01-01 to 2021-08-31**

---

## 🔧 Data Processing

The preprocessing stage includes:

- Datetime conversion
- Extraction of year
- Extraction of month
- Extraction of day
- Extraction of hour
- Extraction of weekday
- Missing-value handling
- Demand percentage calculation
- Preparation of machine-learning features

The original dataset is kept separate from the processed dataset.

---

## 🔍 Exploratory Data Analysis

The EDA stage examines demand patterns according to:

- Hour
- Month
- Year
- Working day
- Weather condition
- Temperature
- Humidity

Important observed patterns include a strong relationship between demand and hour of the day.

The analysis also showed that temperature has a positive relationship with demand, while humidity has a negative relationship with demand.

---

## ⭐ Feature Importance

The Gradient Boosting model was used to analyze feature importance.

The most important features in the current model include:

1. Hour
2. Temperature
3. Year
4. Working day
5. Humidity
6. Weekday
7. Month
8. Feels-like temperature
9. Weather conditions

The current results show that **hour of the day is the dominant prediction factor**.

---

## 🔮 Future Demand Prediction

The project includes a future prediction component.

Users can provide:

- Future date
- Future time
- Holiday status
- Working-day status
- Temperature
- Feels-like temperature
- Humidity
- Weather condition

The application then generates an estimated bike demand.

### Important Limitation

The application does **not** automatically know future weather conditions.

Therefore, the prediction is a:

> **Scenario-based future estimate**

For example, a user can enter a future date and specify an assumed temperature, humidity, and weather condition. The model estimates demand under those assumed conditions.

---

# 🖥️ Streamlit Application

The project includes a small Streamlit application.

The application provides:

### 1. Future Condition Input

Users can select:

- Future date
- Future time
- Holiday
- Working day
- Temperature
- Feels-like temperature
- Humidity
- Weather condition

### 2. Prediction Result

The application displays:

- Predicted demand level
- Estimated bike demand

### 3. Model Information

The sidebar displays:

- Selected model
- MAE
- RMSE
- R²
- Dataset size
- Maximum observed bike count

### 4. Feature Importance

The application displays the important prediction factors from the trained Gradient Boosting model.

### 5. Model Comparison

The application displays the performance of all four trained models.

---

# 📁 Project Structure

```text
bike_demand_predict/
│
├── app/
│   └── app.py
│
├── data/
│   └── original_dataset.csv
│
├── models/
│   ├── linear_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── feature_columns.joblib
│   └── target_column.joblib
│
├── results/
│   ├── 01_loaded_data.csv
│   ├── 02_preprocessed_data.csv
│   ├── X_test.csv
│   ├── y_test.csv
│   ├── model_comparison.csv
│   ├── demand_predictions.csv
│   ├── feature_importance.csv
│   ├── actual_vs_predicted.png
│   ├── model_comparison_r2.png
│   └── feature_importance.png
│
├── src/
│   ├── 01_data_loading.py
│   ├── 02_data_preprocessing.py
│   ├── 03_eda.py
│   ├── 04_model_training.py
│   ├── 05_model_evaluation.py
│   ├── 06_feature_importance.py
│   └── 07_future_prediction.py
│
├── requirements.txt
└── README.md