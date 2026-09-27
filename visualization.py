import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Food_Delivery_Time_Prediction_100.csv")

# 1. Delivery Time Distribution
plt.figure(figsize=(7, 5))
plt.hist(df["Delivery_Time_min"], bins=10)
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Orders")
plt.title("Distribution of Delivery Time")
plt.show()

# 2. Distance vs Delivery Time
plt.figure(figsize=(7, 5))
plt.scatter(df["Distance_km"], df["Delivery_Time_min"])
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.title("Distance vs Delivery Time")
plt.show()

# 3. Traffic Level vs Delivery Time
plt.figure(figsize=(7, 5))
sns.boxplot(
    x="Traffic_Level",
    y="Delivery_Time_min",
    data=df
)
plt.xlabel("Traffic Level")
plt.ylabel("Delivery Time (minutes)")
plt.title("Traffic Level vs Delivery Time")
plt.show()

# 4. Weather vs Delivery Time
plt.figure(figsize=(7, 5))
sns.boxplot(
    x="Weather",
    y="Delivery_Time_min",
    data=df
)
plt.xlabel("Weather")
plt.ylabel("Delivery Time (minutes)")
plt.title("Weather vs Delivery Time")
plt.show()

# 5. Vehicle Type vs Delivery Time
plt.figure(figsize=(7, 5))
sns.boxplot(
    x="Vehicle_Type",
    y="Delivery_Time_min",
    data=df
)
plt.xlabel("Vehicle Type")
plt.ylabel("Delivery Time (minutes)")
plt.title("Vehicle Type vs Delivery Time")
plt.show()
