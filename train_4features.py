import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
df = pd.read_csv("traffic.csv")

# Convert date_time to datetime
df['date_time'] = pd.to_datetime(df['date_time'])

# Extract hour and day_of_week
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek

# Encode weather condition
weather_map = {w: i for i, w in enumerate(df['weather_main'].unique())}
df['weather_encoded'] = df['weather_main'].map(weather_map)

# Encode event/holiday
df['event_val'] = df['holiday'].apply(lambda x: 0 if x == "None" else 1)

# Select features and target
X = df[['hour', 'day_of_week', 'weather_encoded', 'event_val']]
y = df['traffic_volume']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("traffic_model_4features.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained with 4 features and saved!")
