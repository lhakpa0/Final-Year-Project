import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Load the original dataset
df = pd.read_csv("data/Carbon Emission.csv")
print("Initial dataset shape:", df.shape)

# Create total daily screen time feature
df["TotalScreenTime"] = (
    df["How Long TV PC Daily Hour"] +
    df["How Long Internet Daily Hour"]
)

# Scale and combine consumption-related variables
consumption_scaler = MinMaxScaler()
df[["Monthly Grocery Bill", "How Many New Clothes Monthly"]] = consumption_scaler.fit_transform(
    df[["Monthly Grocery Bill", "How Many New Clothes Monthly"]]
)
df["ConsumptionScore"] = (
    df["Monthly Grocery Bill"] +
    df["How Many New Clothes Monthly"]
)

# Convert air travel frequency into numeric form
air_map = {
    "never": 0,
    "rarely": 1,
    "frequently": 2,
    "very frequently": 3
}
df["AirTravelScore"] = df["Frequency of Traveling by Air"].map(air_map)

# Scale and combine travel-related variables
travel_scaler = MinMaxScaler()
df[["Vehicle Monthly Distance Km", "AirTravelScore"]] = travel_scaler.fit_transform(
    df[["Vehicle Monthly Distance Km", "AirTravelScore"]]
)
df["TravelIntensity"] = (
    df["Vehicle Monthly Distance Km"] +
    df["AirTravelScore"]
)

# Create waste-related feature
waste_size_map = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "extra large": 4
}
df["WasteSizeScore"] = df["Waste Bag Size"].map(waste_size_map)
df["WasteScore"] = df["Waste Bag Weekly Count"] * df["WasteSizeScore"]

# List of key engineered features
new_features = [
    "TotalScreenTime",
    "ConsumptionScore",
    "TravelIntensity",
    "WasteScore"
]

# Convert list-like strings into Python lists
df["Recycling"] = df["Recycling"].apply(ast.literal_eval)
recycling_items = list(set(item for sublist in df["Recycling"] for item in sublist))

df["Cooking_With"] = df["Cooking_With"].apply(ast.literal_eval)
cooking_items = list(set(item for sublist in df["Cooking_With"] for item in sublist))

# Create binary columns for recycling categories
for item in recycling_items:
    df[item] = df["Recycling"].apply(lambda x: 1 if item in x else 0)

# Create binary columns for cooking methods
for item in cooking_items:
    df[item] = df["Cooking_With"].apply(lambda x: 1 if item in x else 0)

# Drop original list-based columns
df = df.drop(columns=["Recycling", "Cooking_With"])

print("\nNew columns created:")
print(df[new_features].head())
print(df.shape)

os.makedirs("models", exist_ok=True)

joblib.dump(consumption_scaler, "models/consumption_scaler.pkl")
joblib.dump(travel_scaler, "models/travel_scaler.pkl")
joblib.dump(recycling_items, "models/recycling_items.pkl")
joblib.dump(cooking_items, "models/cooking_items.pkl")

print("Saved preprocessing artifacts to models/")

# Save the engineered dataset
out_path = "data/carbon_engineered.csv"
df.to_csv(out_path, index=False)
print(f"\nNew dataset saved as: {out_path}")

# Display correlation with the target variable
print(df[new_features + ["CarbonEmission"]].corr()["CarbonEmission"])

# Plot correlation values of engineered features
df[new_features + ["CarbonEmission"]].corr()["CarbonEmission"][:-1].plot(kind="bar")
plt.title("Correlation of Engineered Features with Carbon Emission")
plt.ylabel("Correlation value")
plt.tight_layout()
plt.show()