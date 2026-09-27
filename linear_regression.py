import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


# Load encoded dataset
df = pd.read_csv("Food_Delivery_Encoded.csv")


# Features and Target
X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Create Linear Regression Model
model = LinearRegression()


# Train Model
model.fit(X_train, y_train)

print("Linear Regression Model Trained Successfully!")


# Prediction
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\nLinear Regression Evaluation")
print("----------------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))