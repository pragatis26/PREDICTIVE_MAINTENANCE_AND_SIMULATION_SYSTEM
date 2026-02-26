import pickle
import numpy as np
import pandas as pd


# load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


def simulate_conditions(temperature, vibration, pressure, humidity):

    scenarios = []

    # simulate increasing stress
    for t in range(temperature, temperature + 20, 5):

        features = pd.DataFrame([[t, vibration, pressure, humidity]],
                        columns=["temperature", "vibration", "pressure", "humidity"])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        scenarios.append({
            "temperature": t,
            "failure_prediction": int(prediction),
            "failure_probability": round(float(probability), 3)
        })

    return scenarios


# example simulation
if __name__ == "__main__":

    results = simulate_conditions(70, 0.04, 40, 55)

    for r in results:
        print(r)