from pathlib import Path
import joblib
import pandas as pd
from features import PROCESSED_DATA_PATH

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

def main():
    # Load data and model, predict a single row, print result
    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = joblib.load(MODELS_DIR / "best_model.pkl")

    target = "CarbonEmission"
    X = df.drop(columns=[target])
    y = df[target]

    i = 0
    row = X.iloc[[i]]
    actual = y.iloc[i]
    pred = model.predict(row)[0]

    print("Row used:")
    print(row.to_string(index=False))
    print(f"\nActual: {actual}")
    print(f"Predicted: {pred:.2f}")
    print(f"Difference: {pred - actual:.2f}")

if __name__ == "__main__":
    main()
