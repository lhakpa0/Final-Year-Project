import pandas as pd
import matplotlib.pyplot as plt

# Loading dataset
df = pd.read_csv("data/Carbon Emission.csv")

print("Initial dataset shape:", df.shape)

# Feature Engineering Steps
# Feature 1: Total screen time
df["TotalScreenTime"] = (
    df["How Long TV PC Daily Hour"] +
    df["How Long Internet Daily Hour"]
)

# Feature 2: Consumption Score (a composite score based on MonthlyGroceryBill and HowManyNewClothesMonthly)
df["ConsumptionScore"] = (
    df["Monthly Grocery Bill"] +
    df["How Many New Clothes Monthly"] * 50
)

# Feature 3: Travel Score (based on FrequencyTravelAir and VehicleMonthlyDistanceKm)
air_map = {
    "never":0,
    "rarely":1,
    "frequently":2,
    "very frequently":3
}

df["AirTravelScore"] = df["Frequency of Traveling by Air"].map(air_map)

df["TravelIntensity"] = (
    df["Vehicle Monthly Distance Km"] +
    df["AirTravelScore"] * 300
)

# Feature 4: Waste Score (based on WasteBagWeeklyCount and WasteBagSize)
waste_size_map = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "extra large": 4
}

df["WasteSizeScore"] = df["Waste Bag Size"].map(waste_size_map)

df["WasteScore"] = (
    df["Waste Bag Weekly Count"] * df["WasteSizeScore"]
)

new_features = [
    "TotalScreenTime",
    "ConsumptionScore",
    "TravelIntensity",
    "WasteScore"
]

print("\nNew columns created:")
print(df[new_features].head())
print(df.shape)

# Save new dataset to the data directory so other scripts can find it
out_path = "data/carbon_engineered.csv"
df.to_csv(out_path, index=False)
print(f"\nNew dataset saved as: {out_path}")

# Correlation of engineered features with target
print(df[new_features + ["CarbonEmission"]].corr()["CarbonEmission"])

df[new_features + ["CarbonEmission"]].corr()["CarbonEmission"][:-1].plot(kind="bar")
plt.title("Correlation of Engineered Features with Carbon Emission")
plt.ylabel("Correlation value")
plt.show()
