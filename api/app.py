from flask import Flask, request, jsonify
import pickle
import numpy as np

# load model
model_path = "model/model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # expecting JSON input
        data = request.get_json()
        # convert input to correct shape
        features = np.array(data["features"]).reshape(
            1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
