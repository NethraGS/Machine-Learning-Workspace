from flask import Flask, request, jsonify
import joblib
import pandas as pd
from database import init_db, save_to_db, load_db_data

app = Flask(__name__)

model = joblib.load("loan_pretrained.pkl")
init_db()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    result = "Loan Approved" if prediction == 1 else "Loan Rejected"

    save_to_db(data, result)

    return jsonify({"prediction": result})

@app.route("/db-data", methods=["GET"])
def db_data():
    df = load_db_data()
    return df.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
