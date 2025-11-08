import os
# Force TensorFlow to use CPU (important for Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Use absolute paths to avoid file errors on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "house_rent_lstm_model.h5")
scaler_x_path = os.path.join(BASE_DIR, "scaler_x.save")
scaler_y_path = os.path.join(BASE_DIR, "scaler_y.save")

# Load model and scalers
model = load_model(model_path)
scaler_x = joblib.load(scaler_x_path)
scaler_y = joblib.load(scaler_y_path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        features = np.array([
            float(data['bhk']),
            float(data['size']),
            float(data['area_type']),
            float(data['city']),
            float(data['furnishing_status']),
            float(data['tenant_preferred']),
            float(data['bathroom'])
        ]).reshape(1, -1)

        features_scaled = scaler_x.transform(features)
        features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

        prediction_scaled = model.predict(features_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        rent = float(round(prediction[0][0], 2))
        return jsonify({'predicted_rent': rent})

    except Exception as e:
        print("Prediction Error:", str(e))  # Server log
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)










# import streamlit as st
# import numpy as np
# import joblib
# from keras.models import load_model
# import os

# # Force TensorFlow to use CPU (for Streamlit Cloud)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Load model and scalers
# model = load_model("house_rent_lstm_model.h5")
# scaler_x = joblib.load("scaler_x.save")
# scaler_y = joblib.load("scaler_y.save")

# st.set_page_config(page_title="House Rent Price Prediction", page_icon="🏠")

# st.title("🏠 House Rent Price Prediction App")
# st.write("Enter the details below to predict estimated house rent price:")

# # Input fields
# bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
# size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=1000)
# area_type = st.selectbox("Area Type", [0, 1, 2, 3])  # Encode as per training
# city = st.selectbox("City", [0, 1, 2, 3, 4, 5])      # Encode as per training
# furnishing_status = st.selectbox("Furnishing Status", [0, 1, 2])
# tenant_preferred = st.selectbox("Tenant Preferred", [0, 1, 2])
# bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

# if st.button("🔮 Predict Rent"):
#     try:
#         features = np.array([
#             bhk, size, area_type, city, furnishing_status, tenant_preferred, bathroom
#         ]).reshape(1, -1)

#         # Scale input
#         features_scaled = scaler_x.transform(features)
#         features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

#         # Predict
#         prediction_scaled = model.predict(features_scaled)
#         prediction = scaler_y.inverse_transform(prediction_scaled)
#         rent = float(round(prediction[0][0], 2))

#         st.success(f"💰 Estimated Rent: ₹{rent}")
#     except Exception as e:
#         st.error(f"Error: {e}")

