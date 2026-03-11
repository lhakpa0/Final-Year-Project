import pandas as pd

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

print("\nNew columns created:")
print(df[["TotalScreenTime","ConsumptionScore","TravelIntensity","WasteScore"]].head())

# Save new dataset
df.to_csv("carbon_engineered.csv", index=False)

print("\nNew dataset saved as: carbon_engineered.csv")

import pandas as pd
df = pd.read_csv("carbon_engineered.csv")
print(df.shape)
print(df[["TotalScreenTime",
          "ConsumptionScore",
          "TravelIntensity",
          "WasteScore"]].head())

new_features = [
    "TotalScreenTime",
    "ConsumptionScore",
    "TravelIntensity",
    "WasteScore"
]

print(df[new_features + ["CarbonEmission"]].corr()["CarbonEmission"])

import matplotlib.pyplot as plt

df[new_features + ["CarbonEmission"]].corr()["CarbonEmission"][:-1].plot(kind="bar")
plt.title("Correlation of Engineered Features with Carbon Emission")
plt.ylabel("Correlation value")
plt.show()
