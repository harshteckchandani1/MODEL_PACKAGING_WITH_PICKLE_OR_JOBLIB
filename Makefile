# Makefile - Automate ML Packaging Tasks

PYTHON = python
TRAIN_SCRIPT = src/train_model.py
PREDICT_SCRIPT = src/predict_cli.py
MODEL_FILE = models/iris_model.joblib

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train and save model"
	@echo "  make predict    - Run a prediction (example input)"
	@echo "  make clean      - Remove saved model"
	@echo "  make all        - Install + Train + Predict"

install:
	@$(PYTHON) -m pip install -r requirements.txt

train:
	@echo "🚀 Training model..."
	@$(PYTHON) $(TRAIN_SCRIPT)

predict:
	@echo "🔮 Running prediction..."
	@$(PYTHON) $(PREDICT_SCRIPT) 5.1 3.5 1.4 0.2

clean:
	@echo "🧹 Removing model files..."
	@rm -f $(MODEL_FILE)

all: install train predict
