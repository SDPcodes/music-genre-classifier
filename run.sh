#!/bin/bash

set -e

echo "----------------------------------------"
echo "🎼 Music Genre Classifier: Initialization"
echo "----------------------------------------"

# 1. Mac-specific OpenMP check (required for XGBoost)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list libomp &>/dev/null; then
        echo "Missing libomp. Installing via Homebrew..."
        brew install libomp
    fi
fi

# 2. Install Python dependencies (including scikit-learn)
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

# 3. Run Training
echo "[2/3] Training the XGBoost Model..."
if [ ! -d "models" ]; then
  mkdir "models"
fi

if [ ! -d "data" ]; then
  echo "Error: 'data' directory not found. Please ensure CSVs are in the data/ folder."
  exit 1
fi

python train.py

# 4. Start Web App
echo "[3/3] Launching Web Application..."
streamlit run app.py
