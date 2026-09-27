import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# Create Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train Model
rf_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")


# Prediction
rf_pred = rf_model.predict(X_test)


# Evaluation
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_pred)
)

rf_r2 = r2_score(y_test, rf_pred)


print("\nRandom Forest Evaluation")
print("------------------------")
print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 2))