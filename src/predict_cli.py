# src/predict_cli.py
import sys
from joblib import load
import os

USAGE = "Usage: python predict_cli.py <f1> <f2> <f3> <f4>"

if len(sys.argv) != 5:
    print(USAGE)
    sys.exit(1)

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'iris_model.joblib')

if not os.path.exists(model_path):
    print("❌ Model file not found. Run 'python src/train_model.py' or 'make train' first.")
    sys.exit(1)

try:
    features = [float(x) for x in sys.argv[1:5]]
except ValueError:
    print("❌ All features must be numeric.")
    print(USAGE)
    sys.exit(1)

model = load(model_path)
prediction = model.predict([features])
print(f"✅ Predicted class: {prediction[0]}")
