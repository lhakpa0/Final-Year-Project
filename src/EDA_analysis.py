import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
raw_path = "Carbon Emission.csv"
df = pd.read_csv(raw_path)
TARGET = "CarbonEmission"

if TARGET in df.columns:
    print("\n=== Target Variable Summary ===")
    print(df[TARGET].describe())

    # Target distribution
    plt.figure()
    df[TARGET].hist(bins=30)
    plt.title("Distribution of CarbonEmission Target")
    plt.xlabel("CarbonEmission")
    plt.ylabel("Frequency")
    plt.show()

print("Shape:", df.shape)
print(df.head())

# 1) Initial data exploration
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

# 2)REMOVE DUPLICATES
df = df.drop_duplicates()

# 3)Removing rows where the target is missing.
df = df.dropna(subset=[TARGET])











