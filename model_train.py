# model_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib

# Load data
data = pd.read_csv("House_Rent_Dataset.csv")

# Map categorical columns to numeric
data["Area Type"] = data["Area Type"].map({"Super Area": 1, "Carpet Area": 2, "Built Area": 3})
data["City"] = data["City"].map({
    "Mumbai": 4000, "Chennai": 6000, "Bangalore": 5600,
    "Hyderabad": 5000, "Delhi": 1100, "Kolkata": 7000
})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2})
data["Tenant Preferred"] = data["Tenant Preferred"].map({
    "Bachelors/Family": 2, "Bachelors": 1, "Family": 3
})

# Features and target
X = data[["BHK", "Size", "Area Type", "City", "Furnishing Status", "Tenant Preferred", "Bathroom"]].values
y = data[["Rent"]].values

# Scale features and target
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.10, random_state=42)

# Reshape for LSTM (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=21, batch_size=1)

# Save model and scalers
model.save("house_rent_lstm_model.h5")
joblib.dump(scaler_x, "scaler_x.save")
joblib.dump(scaler_y, "scaler_y.save")

print("Model and scalers saved successfully.")
