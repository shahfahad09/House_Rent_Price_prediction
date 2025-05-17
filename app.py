# app.py
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scalers once at startup
model = load_model("house_rent_lstm_model.h5")
scaler_x = joblib.load("scaler_x.save")
scaler_y = joblib.load("scaler_y.save")

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

        rent = float(round(prediction[0][0], 2))  # <-- yahan float conversion

        return jsonify({'predicted_rent': rent})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
