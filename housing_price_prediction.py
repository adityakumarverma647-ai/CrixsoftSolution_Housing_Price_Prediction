# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

print(df.head())

# Dataset shape

print(df.shape)

# Column names

print(df.columns)

# Dataset info

df.info()

# Statistical summary

print(df.describe())

# Check missing values

print(df.isnull().sum())

# Histogram

df.hist(figsize=(12,10))
plt.show()

# Correlation heatmap

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.show()

# Scatter plot

plt.figure(figsize=(8,5))

plt.scatter(
    df["MedInc"],
    df["Price"]
)

plt.xlabel("Median Income")
plt.ylabel("House Price")

plt.show()

# Features and target

X = df.drop("Price", axis=1)
y = df["Price"]

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model

model = LinearRegression()

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

print(y_pred[:10])

# Model evaluation

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Actual vs predicted

plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.show()

# Conclusion

print("Housing price prediction completed successfully.")