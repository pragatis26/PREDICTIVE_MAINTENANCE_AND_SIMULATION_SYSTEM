import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
data = pd.read_csv("data/sample_sensor_data.csv")

# Features and target
X = data.drop("failure", axis=1)
y = data["failure"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save trained model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model/model.pkl")