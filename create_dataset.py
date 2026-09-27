import pandas as pd
import numpy as np

np.random.seed(42)

n = 100

weather = np.random.choice(
    ["Clear", "Cloudy", "Rainy", "Foggy"], n
)

traffic = np.random.choice(
    ["Low", "Medium", "High"], n
)

vehicle = np.random.choice(
    ["Motorcycle", "Scooter", "Car"], n
)

order_type = np.random.choice(
    ["Meal", "Snacks", "Drinks", "Dessert"], n
)

time_of_day = np.random.choice(
    ["Morning", "Afternoon", "Evening", "Night"], n
)

festival = np.random.choice(
    ["No", "Yes"], n, p=[0.8, 0.2]
)

df = pd.DataFrame({
    "Age": np.random.randint(20, 51, n),
    "Rating": np.round(np.random.uniform(3.0, 5.0, n), 1),
    "Distance_km": np.round(np.random.uniform(1, 15, n), 1),
    "Weather": weather,
    "Traffic_Level": traffic,
    "Vehicle_Type": vehicle,
    "Order_Type": order_type,
    "Time_of_Day": time_of_day,
    "Festival": festival,
    "Multiple_Deliveries": np.random.randint(0, 3, n)
})

df["Delivery_Time_min"] = np.round(
    12
    + df["Distance_km"] * 3.2
    + df["Weather"].map({
        "Clear": 0,
        "Cloudy": 2,
        "Rainy": 7,
        "Foggy": 5
    })
    + df["Traffic_Level"].map({
        "Low": 0,
        "Medium": 6,
        "High": 13
    })
    + df["Vehicle_Type"].map({
        "Motorcycle": 0,
        "Scooter": 2,
        "Car": 4
    })
    + df["Time_of_Day"].map({
        "Morning": 0,
        "Afternoon": 2,
        "Evening": 5,
        "Night": 3
    })
    + (df["Festival"] == "Yes") * 6
    + df["Multiple_Deliveries"] * 5
    + (5 - df["Rating"]) * 2
    + np.random.normal(0, 3, n),
    1
)

df["Delivery_Time_min"] = df["Delivery_Time_min"].clip(lower=10)

df.to_csv(
    "Food_Delivery_Time_Prediction_100.csv",
    index=False
)

print("Dataset created successfully!")
print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

print("\nFirst 5 Records:")
print(df.head())
