import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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


# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)


# Save model
joblib.dump(model, "food_delivery_model.pkl")


# Save feature columns
joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)


print("Model saved successfully!")
print("food_delivery_model.pkl created!")
print("feature_columns.pkl created!")