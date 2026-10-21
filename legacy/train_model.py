import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Example dataset: symptoms and diseases
data = {
    "Fever": [1, 0, 1, 0, 1],
    "Cough": [1, 0, 1, 1, 0],
    "Headache": [0, 1, 1, 1, 1],
    "Fatigue": [1, 0, 1, 0, 1],
    "Shortness of breath": [0, 1, 0, 1, 1],
    "Sore throat": [1, 1, 0, 0, 1],
    "Runny nose": [1, 0, 0, 1, 0],
    "Chest pain": [0, 1, 0, 1, 0],
    "Muscle pain": [1, 0, 1, 1, 0],
    "Nausea": [0, 1, 0, 1, 1],
    "Dizziness": [1, 0, 1, 0, 1],
    "Loss of taste": [0, 1, 0, 0, 1],
    "Disease": ["Flu", "Migraine", "Flu", "Asthma", "COVID-19"]
}

df = pd.DataFrame(data)

# Features and labels
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as disease_model.pkl")

