from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load deployment-safe artifacts
model = joblib.load("deployment/artifacts/linear_reg_model.joblib")
feature_order = joblib.load("deployment/artifacts/feature_order.joblib")

REQUIRED_FIELDS = ["pH","temperature_C","chloride_ppm","dissolved_oxygen_mgL","flow_velocity_mps"]

@app.get("/")
def home():
  return "✅ Corrosion Linear Regression API is running"

@app.get("/health")
def health():
  return jsonify(status="ok")

@app.post("/predict")
def predict():
  data = request.get_json(silent=True)

  if not data:
      return jsonify(error="Invalid or missing JSON body"), 400

  missing = [k for k in REQUIRED_FIELDS if k not in data]
  if missing:
      return jsonify(error=f"Missing fields: {missing}", required=REQUIRED_FIELDS), 400

  try:
      row = {k: float(data[k]) for k in REQUIRED_FIELDS}
      row["pH_squared"] = row["pH"] ** 2  # feature engineering at inference

      X = pd.DataFrame([row])[feature_order]
      yhat = model.predict(X)[0]

      return jsonify(predicted_corrosion_rate_mm_per_yr=float(yhat))
  except Exception as e:
      return jsonify(error=str(e)), 500

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)

