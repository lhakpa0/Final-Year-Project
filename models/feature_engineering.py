import ast
import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from utils import apply_basic_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Load the original dataset
df = pd.read_csv(os.path.join(BASE_DIR, "data", "Carbon Emission.csv"))
print("Initial dataset shape:", df.shape)

# Convert list-like strings into actual lists
df["Recycling"] = df["Recycling"].apply(ast.literal_eval)
df["Cooking_With"] = df["Cooking_With"].apply(ast.literal_eval)

# Get unique items - sorted so columns are always in the same order
recycling_items = sorted(set(item for sublist in df["Recycling"] for item in sublist))
cooking_items = sorted(set(item for sublist in df["Cooking_With"] for item in sublist))

# Binary encode recycling and cooking columns
for item in recycling_items:
    df[item] = df["Recycling"].apply(lambda x, i=item: 1 if i in x else 0)

for item in cooking_items:
    df[item] = df["Cooking_With"].apply(lambda x, i=item: 1 if i in x else 0)

df = df.drop(columns=["Recycling", "Cooking_With"])

# Add features that don't need scaling
# ConsumptionScore needs a scaler, so that's done in model_compare.py after the train/test split to avoid data leakage)
df = apply_basic_features(df)

print("\nNew columns created:")
print(df[["WasteScore"]].head())
print(df.shape)

# Save items lists for the app
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(recycling_items, os.path.join(BASE_DIR, "models", "recycling_items.pkl"))
joblib.dump(cooking_items, os.path.join(BASE_DIR, "models", "cooking_items.pkl"))
print("Saved item lists to models/")

# Save engineered dataset
out_path = os.path.join(BASE_DIR, "data", "carbon_engineered.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Quick correlation check
print(df[["WasteScore", "CarbonEmission"]].corr()["CarbonEmission"])

df[["WasteScore", "CarbonEmission"]].corr()["CarbonEmission"][:-1].plot(kind="bar")
plt.title("Correlation of Engineered Features with Carbon Emission")
plt.ylabel("Correlation value")
plt.tight_layout()
plt.show()
