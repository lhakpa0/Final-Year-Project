import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# 1) Initial data exploration
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

# 2)REMOVE DUPLICATES
df = df.drop_duplicates()

# 3)Removing rows where the target is missing.
df = df.dropna(subset=["CarbonEmission"])

# Separate Features and Target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Ordinal Encoding (Ordered Categories)
ordinal_mappings = {
    "HowOftenShower": {
        "Rarely": 0,
        "Sometimes": 1,
        "Daily": 2
    },

    "FrequencyTravelAir": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Frequently": 3,
        "Very Frequently": 4
    },

    "WasteBagSize": {
        "Small": 0,
        "Medium": 1,
        "Large": 2,
        "Extra Large": 3
    },

    "EnergyEfficiency": {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }
}

# Applying ordinal mappings
for col, mapping in ordinal_mappings.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)


# Nominal Encoding (One-Hot Encoding)
# Identifying nominal categorical columns automatically
nominal_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNominal columns to One-Hot Encode:")
print(nominal_cols)

# Applying One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

# Saving Encoded Dataset
X_encoded.to_csv("X_encoded.csv", index=False)
y.to_csv("y_target.csv", index=False)

print("\nSaved encoded features to: X_encoded.csv")
print("Saved target to: y_target.csv")

# Loading encoded data
X_encoded = pd.read_csv("X_encoded.csv")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Training a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

print(f"Carbon Emission Range: {y.min()} to {y.max()}")
print(f"Mean Carbon Emission: {y.mean()}")
print(f"MAE as % of mean: {(230.370 / y.mean()) * 100:.2f}%")

# feature importance
importances = model.feature_importances_
feature_names = X_encoded.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(importance_df.head(10))

# Visualizing
plt.figure(figsize=(10, 6))
plt.barh(importance_df.head(10)['Feature'], importance_df.head(10)['Importance'])
plt.xlabel('Importance')
plt.title('Top 10 Features Affecting Carbon Emissions')
plt.gca().invert_yaxis()
plt.show()
