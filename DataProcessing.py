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

df = pd.read_csv(raw_path)

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())

# Removing rows where the target is missing.
df = df.dropna(subset=["CarbonEmission"])

# Separating features (X) and target (y)
X = df.drop("CarbonEmission", axis=1)
y = df["CarbonEmission"]

