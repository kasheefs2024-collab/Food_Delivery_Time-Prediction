import pandas as pd

# Load Dataset
df = pd.read_csv("Food_Delivery_Time_Prediction_100.csv")

# Convert categorical columns into numerical columns
df_encoded = pd.get_dummies(
    df,
    columns=[
        "Weather",
        "Traffic_Level",
        "Vehicle_Type",
        "Order_Type",
        "Time_of_Day",
        "Festival"
    ],
    drop_first=True
)

print("ENCODED DATASET")
print(df_encoded.head())

print("\nCOLUMNS AFTER ENCODING")
print(df_encoded.columns.tolist())

print("\nDATASET SHAPE")
print(df_encoded.shape)

# Save encoded dataset
df_encoded.to_csv(
    "Food_Delivery_Encoded.csv",
    index=False
)

print("\nEncoded dataset saved successfully!")