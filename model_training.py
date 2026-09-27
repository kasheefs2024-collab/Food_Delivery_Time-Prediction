import pandas as pd
from sklearn.model_selection import train_test_split

# Load encoded dataset
df = pd.read_csv("Food_Delivery_Encoded.csv")

# Target column
y = df["Delivery_Time_min"]

# Features
X = df.drop("Delivery_Time_min", axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("FEATURES SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)

print("\nTRAINING DATA:")
print(X_train.shape)

print("\nTESTING DATA:")
print(X_test.shape)