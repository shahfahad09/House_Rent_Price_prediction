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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)




# # app.py
# from flask import Flask, request, jsonify, render_template
# from keras.models import load_model
# import numpy as np
# import joblib
# import os

# app = Flask(__name__)

# # Load model and scalers once at startup
# model = load_model("house_rent_lstm_model.h5")
# scaler_x = joblib.load("scaler_x.save")
# scaler_y = joblib.load("scaler_y.save")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form
#     try:
#         features = np.array([
#             float(data['bhk']),
#             float(data['size']),
#             float(data['area_type']),
#             float(data['city']),
#             float(data['furnishing_status']),
#             float(data['tenant_preferred']),
#             float(data['bathroom'])
#         ]).reshape(1, -1)

#         features_scaled = scaler_x.transform(features)
#         features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

#         prediction_scaled = model.predict(features_scaled)
#         prediction = scaler_y.inverse_transform(prediction_scaled)

#         rent = float(round(prediction[0][0], 2))

#         return jsonify({'predicted_rent': rent})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# #  Updated for render deployment
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)





# #app.py
# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from flask import Flask, request, jsonify, render_template
# from keras.models import load_model
# import numpy as np
# import joblib


# app = Flask(__name__)

# # Absolute path setup to avoid file loading issues on Render
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model_path = os.path.join(BASE_DIR, "house_rent_lstm_model.h5")
# scaler_x_path = os.path.join(BASE_DIR, "scaler_x.save")
# scaler_y_path = os.path.join(BASE_DIR, "scaler_y.save")

# # Load model and scalers once at startup
# model = load_model("house_rent_lstm_model.h5")
# scaler_x = joblib.load("scaler_x.save")
# scaler_y = joblib.load("scaler_y.save")


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form
#     try:
#         # Get input features from form
#         features = np.array([
#             float(data['bhk']),
#             float(data['size']),
#             float(data['area_type']),
#             float(data['city']),
#             float(data['furnishing_status']),
#             float(data['tenant_preferred']),
#             float(data['bathroom'])
#         ]).reshape(1, -1)

#         # Preprocess input
#         features_scaled = scaler_x.transform(features)
#         features_scaled = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

#         # Predict
#         prediction_scaled = model.predict(features_scaled)
#         prediction = scaler_y.inverse_transform(prediction_scaled)

#         rent = float(round(prediction[0][0], 2))

#         return jsonify({'predicted_rent': rent})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# # Required for Render deployment
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)


