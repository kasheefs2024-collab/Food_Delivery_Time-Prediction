import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Food_Delivery_Time_Prediction_100.csv")

# First 5 records
print("FIRST 5 RECORDS")
print(df.head())

# Dataset information
print("\nDATASET INFORMATION")
print(df.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())