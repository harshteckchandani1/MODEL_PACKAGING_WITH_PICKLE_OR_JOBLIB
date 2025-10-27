# src/train_model.py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import os

# Ensure models directory exists (relative path from src/)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'iris_model.joblib')

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
dump(model, model_path)
print(f"✅ Model trained and saved in {model_path}")
