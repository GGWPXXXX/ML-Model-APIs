from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import sys

# Initialize Flask app
app = Flask(__name__)

# Load the model and columns at startup
try:
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_columns = None


@app.route('/')
def home():
    return "Welcome to the ML Model API! Use the /predict endpoint to make predictions."


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Train and load the model first.'}), 500

    try:
        json_ = request.get_json()
        if not json_:
            return jsonify({'error': 'Empty request body'}), 400

        print(f"Received input: {json_}")
        query_df = pd.DataFrame(json_)
        query_df = pd.get_dummies(query_df)
        query_df = query_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(query_df).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except (IndexError, ValueError):
        port = 12345  # Default port

    app.run(port=port, debug=True)
