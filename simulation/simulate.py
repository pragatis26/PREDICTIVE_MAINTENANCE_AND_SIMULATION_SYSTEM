import pandas as pd
import joblib

model = joblib.load("../model/rf_model.pkl")

def simulate(sensor_data, temp_increase=0, vibration_increase=0):
    data = sensor_data.copy()
    data['temperature'] += temp_increase
    data['vibration'] += vibration_increase
    predictions = model.predict_proba(data)[:,1]
    return predictions