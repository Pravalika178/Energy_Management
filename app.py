# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Title
st.title("Smart Energy Consumption Predictor")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("forecast_model.pkl")

model = load_model()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("household_energy.csv", parse_dates=["timestamp"])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek
    return df

df = load_data()

# Show data
if st.checkbox("Show Raw Dataset"):
    st.write(df.head())

# Line chart
st.subheader("Energy Consumption Over Time")
st.line_chart(df.set_index("timestamp")["energy_consumption"])

# Prediction Input
st.subheader("Predict Energy Consumption")

col1, col2 = st.columns(2)

with col1:
    temp = st.slider("Indoor Temperature (°C)", 15.0, 35.0, 25.0)
    device = st.selectbox("Device Usage (On=1 / Off=0)", [1, 0])

with col2:
    out_temp = st.slider("Outside Temperature (°C)", 20.0, 45.0, 30.0)
    hour = st.slider("Hour of Day", 0, 23, 12)

day_name_to_num = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
day = st.selectbox("Day of Week", list(day_name_to_num.keys()))
day_num = day_name_to_num[day]

# Prepare input
input_data = np.array([[temp, out_temp, device, hour, day_num]])

# Predict
prediction = model.predict(input_data)[0]
st.success(f"Predicted Energy Consumption: **{round(prediction, 2)} kWh**")