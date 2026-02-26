from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Predictive Maintenance API")

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}


@app.post("/predict")
def predict_failure(temperature: float, vibration: float, pressure: float, humidity: float):

    # Convert input to model format
    features = np.array([[temperature, vibration, pressure, humidity]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "failure_prediction": int(prediction),
        "failure_probability": float(round(probability, 3))
    }
    
@app.get("/simulate")
def simulate_conditions(
    temperature: float,
    vibration: float,
    pressure: float,
    humidity: float
):

    results = []

    for t in range(int(temperature), int(temperature) + 20, 5):

        features = pd.DataFrame(
            [[t, vibration, pressure, humidity]],
            columns=["temperature", "vibration", "pressure", "humidity"]
        )

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        results.append({
            "temperature": t,
            "failure_prediction": int(prediction),
            "failure_probability": round(float(probability), 3)
        })

    return {
        "simulation_results": results
    }