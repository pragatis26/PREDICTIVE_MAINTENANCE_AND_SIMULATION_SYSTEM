from fastapi import FastAPI
import pandas as pd
from simulation.simulate import simulate

app = FastAPI()

@app.post("/predict")
def predict(sensor_data: dict):
    df = pd.DataFrame([sensor_data])
    failure_prob = simulate(df)[0]
    return {"failure_probability": round(failure_prob, 2)}