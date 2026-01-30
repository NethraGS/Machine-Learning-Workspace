from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

linear_model = joblib.load("./models/linear_model.pkl")
logistic_model = joblib.load("./models/logistic_model.pkl")

FEATURE_COLUMNS = [
    "date",
    "humidity_percent",
    "pressure_hpa",
    "wind_speed_kmph",
    "cloud_cover_percent",
    "rainfall_mm",
    "sunshine_hours",
    "temperature_c"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        linear_pred = float(linear_model.predict(df)[0])
        logistic_pred = int(logistic_model.predict(df)[0])

        return jsonify({
            "linear_prediction": linear_pred,
            "logistic_prediction": logistic_pred
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
