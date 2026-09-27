import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("food_delivery_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Page Configuration
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔"
)

# Title
st.title("🍔 Food Delivery Time Prediction")
st.write("Enter the order details to predict delivery time.")

# User Inputs
age = st.number_input(
    "Delivery Partner Age",
    min_value=18,
    max_value=60,
    value=25
)

rating = st.number_input(
    "Delivery Partner Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.5,
    step=0.1
)

distance = st.number_input(
    "Distance (km)",
    min_value=0.1,
    max_value=30.0,
    value=5.0,
    step=0.1
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Cloudy", "Rainy", "Foggy"]
)

traffic = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

vehicle = st.selectbox(
    "Vehicle Type",
    ["Motorcycle", "Scooter", "Car"]
)

order_type = st.selectbox(
    "Order Type",
    ["Meal", "Snacks", "Drinks", "Dessert"]
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

festival = st.selectbox(
    "Festival",
    ["No", "Yes"]
)

multiple_deliveries = st.number_input(
    "Multiple Deliveries",
    min_value=0,
    max_value=2,
    value=0
)

# Prediction Button
if st.button("🚀 Predict Delivery Time"):

    new_order = pd.DataFrame({
        "Age": [age],
        "Rating": [rating],
        "Distance_km": [distance],
        "Weather": [weather],
        "Traffic_Level": [traffic],
        "Vehicle_Type": [vehicle],
        "Order_Type": [order_type],
        "Time_of_Day": [time_of_day],
        "Festival": [festival],
        "Multiple_Deliveries": [multiple_deliveries]
    })

    # Convert categorical data
    new_order_encoded = pd.get_dummies(
        new_order,
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

    # Match training columns
    new_order_encoded = new_order_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(new_order_encoded)[0]

    st.success(
        f"🍔 Estimated Delivery Time: {prediction:.1f} minutes"
    )