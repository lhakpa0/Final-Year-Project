import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the raw dataset
raw_path = "data/Carbon Emission.csv"
df = pd.read_csv(raw_path)
TARGET = "CarbonEmission"

# Display summary statistics for the target variable
if TARGET in df.columns:
    print("\n=== Target Variable Summary ===")
    print(df[TARGET].describe())

    # Plot the distribution of the target variable
    plt.figure()
    df[TARGET].hist(bins=30)
    plt.title("Distribution of CarbonEmission Target")
    plt.xlabel("CarbonEmission")
    plt.ylabel("Frequency")
    plt.show()

# Display dataset shape and first few rows
print("Shape:", df.shape)
print(df.head())

# Check for missing values and duplicate rows
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where the target value is missing
df = df.dropna(subset=[TARGET])