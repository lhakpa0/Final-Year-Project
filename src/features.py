import ast
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Carbon Emission.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "carbon_engineered.csv"
MODELS_DIR = BASE_DIR / "models"

# Map bag sizes to numbers for WasteScore calculation
WASTE_SIZE_MAP = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "extra large": 4,
}

# These features are excluded from SHAP charts because those are demographic features and users cannot change them.
DEMOGRAPHIC_FEATURES = {"Sex", "Body Type"}

def apply_basic_features(df):
    # Create waste score
    df = df.copy()
    df["WasteSizeScore"] = df["Waste Bag Size"].map(WASTE_SIZE_MAP) # Map bag sizes to numeric scores based on the defined mapping, which allows us to combine it with the count of bags for a more meaningful feature.
    df["WasteScore"] = df["Waste Bag Weekly Count"] * df["WasteSizeScore"]
    return df.drop(columns=["WasteSizeScore"])

def apply_scaled_features(df, consumption_scaler):
    # Combine shopping features into consumption score
    df = df.copy()
    scaled = consumption_scaler.transform(
        df[["Monthly Grocery Bill", "How Many New Clothes Monthly"]]
    )
    df["ConsumptionScore"] = scaled[:, 0] + scaled[:, 1] # simple sum of scaled features, could also try weighted sum or PCA for more complex combinations  
    return df

def build_engineered_dataset():
    # Load raw CSV, parse multi select columns, one hot encode them, and add WasteScore
    df = pd.read_csv(RAW_DATA_PATH)
    print("Initial dataset shape:", df.shape)

    df["Recycling"] = df["Recycling"].apply(ast.literal_eval)
    df["Cooking_With"] = df["Cooking_With"].apply(ast.literal_eval)

    recycling_items = sorted(set(item for sublist in df["Recycling"] for item in sublist))
    cooking_items = sorted(set(item for sublist in df["Cooking_With"] for item in sublist))

    for item in recycling_items:
        df[item] = df["Recycling"].apply(lambda x, i=item: 1 if i in x else 0)

    for item in cooking_items:
        df[item] = df["Cooking_With"].apply(lambda x, i=item: 1 if i in x else 0)

    df = df.drop(columns=["Recycling", "Cooking_With"])
    df = apply_basic_features(df)
    return df, recycling_items, cooking_items

def save_feature_artifacts(df, recycling_items, cooking_items):
    # Save processed data and item lists so train.py can use them
    MODELS_DIR.mkdir(exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(recycling_items, MODELS_DIR / "recycling_items.pkl")
    joblib.dump(cooking_items, MODELS_DIR / "cooking_items.pkl")
    print("Saved item lists to models/")

    df.to_csv(PROCESSED_DATA_PATH, index=False)  
    print(f"\nSaved: {PROCESSED_DATA_PATH}")

def plot_feature_check(df):
    # Quick correlation check to verify engineered features are useful
    print(df[["WasteScore", "CarbonEmission"]].corr()["CarbonEmission"])

    df[["WasteScore", "CarbonEmission"]].corr()["CarbonEmission"][:-1].plot(kind="bar")
    plt.title("Correlation of Engineered Features with Carbon Emission")
    plt.ylabel("Correlation value")
    plt.tight_layout()
    plt.show()

def main(): 
    # Run the full feature engineering pipeline
    df, recycling_items, cooking_items = build_engineered_dataset()
    print("\nNew columns created:")
    print(df[["WasteScore"]].head())
    print(df.shape)

    save_feature_artifacts(df, recycling_items, cooking_items)
    plot_feature_check(df) #

if __name__ == "__main__":
    main()
