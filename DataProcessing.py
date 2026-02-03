import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split#
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
raw_path = "Carbon Emission.csv"

columns = [
    "BodyType", "Sex", "Diet", "HowOftenShower", "HeatingEnergySource",
    "Transport", "VehicleType", "SocialActivity", "MonthlyGroceryBill",
    "FrequencyTravelAir", "VehicleMonthlyDistanceKm", "WasteBagSize",
    "WasteBagWeeklyCount", "HowLongTVPCDailyHour", "HowManyNewClothesMonthly",
    "HowLongInternetDailyHour", "EnergyEfficiency", "Recycling",
    "CookingWith", "CarbonEmission"
]

TARGET = "CarbonEmission"
df = pd.read_csv(raw_path)

df.columns = columns
# Defining categorical and numerical columns
CATEGORICAL_COLS = [
    "BodyType", "Sex", "Diet", "HowOftenShower", "HeatingEnergySource",
    "Transport", "VehicleType", "SocialActivity", "WasteBagSize",
    "Recycling", "CookingWith"
]
NUMERIC_COLS = [
    "MonthlyGroceryBill",
    "VehicleMonthlyDistanceKm",
    "WasteBagWeeklyCount",
    "HowLongTVPCDailyHour",
    "HowManyNewClothesMonthly",
    "HowLongInternetDailyHour"
]

# Initial data exploration
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))

print("\nDuplicate rows:", df.duplicated().sum())

# Optional: remove duplicates
df = df.drop_duplicates()

# Removing rows where the target is missing.
df = df.dropna(subset=["CarbonEmission"])

# Separating features (X) and target (y)
X = df.drop("CarbonEmission", axis=1)
y = df["CarbonEmission"]

