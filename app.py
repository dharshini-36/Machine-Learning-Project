import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("traffic_model_4features.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚦 Traffic Flow Prediction System (4 Features)")
st.write("Predict traffic volume using Multiple Linear Regression")

# Inputs
hour = st.selectbox("Hour of Day", list(range(24)))

day_map = {
    "Monday":0, "Tuesday":1, "Wednesday":2,
    "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6
}
day_name = st.selectbox("Day of Week", list(day_map.keys()))
day_of_week = day_map[day_name]

weather_map = {"Clear":0, "Clouds":1, "Rain":2, "Snow":3, "Fog":4}  # adjust based on dataset
weather = st.selectbox("Weather Condition", list(weather_map.keys()))
weather_encoded = weather_map[weather]

event = st.radio("Nearby Event Occurring?", ["No","Yes"])
event_val = 1 if event=="Yes" else 0

# Predict
if st.button("Predict Traffic Volume"):
    input_data = np.array([[hour, day_of_week, weather_encoded, event_val]])
    prediction = model.predict(input_data)
    traffic_volume = int(prediction[0])
    
    # Determine traffic flow
    if traffic_volume <= 2000:
        traffic_flow = "No/Light Traffic"
    elif traffic_volume <= 4000:
        traffic_flow = "Moderate Traffic"
    else:
        traffic_flow = "Heavy Traffic"
    
    st.success(f"🚗 Predicted Traffic Volume: {traffic_volume} vehicles/hour")
    st.info(f"🛣 Traffic Flow: {traffic_flow}")
