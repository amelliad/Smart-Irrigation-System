from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# =====================================
# LOAD MODEL
# =====================================

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(
    BASE_DIR,
    'model',
    'rf_irrigation_model.pkl'
)

scaler_path = os.path.join(
    BASE_DIR,
    'model',
    'scaler.pkl'
)

encoder_path = os.path.join(
    BASE_DIR,
    'model',
    'label_encoders.pkl'
)

# =====================================
# LOAD FILE
# =====================================

model = joblib.load(model_path)

scaler = joblib.load(scaler_path)

label_encoders = joblib.load(encoder_path)

# =====================================
# FLASK APP
# =====================================

app = Flask(__name__)

# =====================================
# HOME ROUTE
# =====================================

@app.route('/')
def home():

    return jsonify({

        'message': 'Smart Irrigation API Running'

    })

# =====================================
# PREDICT ROUTE
# =====================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # =========================
        # GET JSON DATA
        # =========================

        data = request.json

        # =========================
        # CREATE DATAFRAME
        # =========================

        df = pd.DataFrame([{

            'Soil_Moisture':
            data['Soil_Moisture'],

            'Temperature_C':
            data['Temperature_C'],

            'Humidity':
            data['Humidity'],

            'Rainfall_mm':
            data['Rainfall_mm'],

            'Sunlight_Hours':
            data['Sunlight_Hours'],

            'Crop_Growth_Stage':
            data['Crop_Growth_Stage'],

            'Mulching_Used':
            data['Mulching_Used'],

            'Wind_Speed_kmh':
            data['Wind_Speed_kmh']

        }])

        # =========================
        # SCALING
        # =========================

        scaled = scaler.transform(df)

        # =========================
        # PREDICT
        # =========================

        prediction = model.predict(scaled)

        # =========================
        # DECODE LABEL
        # =========================

        result = label_encoders[
            'Irrigation_Need'
        ].inverse_transform(prediction)

        # =========================
        # RESPONSE
        # =========================

        return jsonify({

            'prediction': str(result[0])

        })

    except Exception as e:

        return jsonify({

            'error': str(e)

        }), 500

# =====================================
# RUN APP
# =====================================

if __name__ == '__main__':

    app.run(

        host='0.0.0.0',
        port=5000

    )
