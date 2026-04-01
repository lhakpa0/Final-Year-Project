import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(BASE_DIR, "data", "Carbon Emission.csv")
TARGET = "CarbonEmission"
PLOT_DIR = os.path.join(BASE_DIR, "src", "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(raw_path)

#  Dataset overview 

print("Dataset overview")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Missing values

print("\nMissing values & duplicates")

missing = df.isna().sum()
print(f"Missing values:")
print(missing[missing > 0] if missing.any() else "None")
print(f"\nTotal missing: {missing.sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# Vehicle Type is only filled in when Transport = private, so those NaNs
# are expected is not actual missing data
if "Vehicle Type" in df.columns:
    print("\nVehicle Type by Transport mode:")
    cross = pd.crosstab(df["Transport"], df["Vehicle Type"].isna(), margins=True)
    cross.columns = ["Has Value", "NaN", "Total"]
    print(cross)

# Target variable

print("\nTarget variable")
print(df[TARGET].describe())
print(f"\nSkewness: {df[TARGET].skew():.3f}")
print(f"Kurtosis: {df[TARGET].kurtosis():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[TARGET], bins=40, edgecolor="white", alpha=0.8)
axes[0].axvline(df[TARGET].mean(), color="red", linestyle="--", label=f"Mean: {df[TARGET].mean():.0f}")
axes[0].axvline(df[TARGET].median(), color="orange", linestyle="--", label=f"Median: {df[TARGET].median():.0f}")
axes[0].set_xlabel("CarbonEmission (kg CO₂e/month)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Carbon Emission")
axes[0].legend()

axes[1].boxplot(df[TARGET], vert=True)
axes[1].set_ylabel("CarbonEmission")
axes[1].set_title("Box Plot")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/target_distribution.png", dpi=150)
plt.close()

# Numeric features

print("\nNumeric features")
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET)
print(df[numeric_cols].describe().round(2))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(numeric_cols):
    ax = axes.flat[i]
    ax.hist(df[col], bins=30, edgecolor="white", alpha=0.8)
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("Frequency")
for j in range(len(numeric_cols), len(axes.flat)):
    axes.flat[j].set_visible(False)
plt.suptitle("Numeric Feature Distributions", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/numeric_distributions.png", dpi=150)
plt.close()
# Categorical features

print("\nCategorical features")

cat_cols = ["Body Type", "Sex", "Diet", "How Often Shower",
            "Heating Energy Source", "Transport", "Vehicle Type",
            "Social Activity", "Frequency of Traveling by Air",
            "Waste Bag Size", "Energy efficiency"]

for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
for i, col in enumerate(cat_cols):
    ax = axes.flat[i]
    df[col].value_counts(dropna=False).plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_title(col, fontsize=10)
    ax.tick_params(axis="x", rotation=45)
for j in range(len(cat_cols), len(axes.flat)):
    axes.flat[j].set_visible(False)
plt.suptitle("Categorical Feature Distributions", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/categorical_distributions.png", dpi=150)
plt.close()

# Emissions by category

print("\nEmissions by category")

key_cats = ["Diet", "Transport", "Heating Energy Source", "Body Type",
            "Frequency of Traveling by Air", "Energy efficiency"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(key_cats):
    ax = axes.flat[i]
    order = df.groupby(col)[TARGET].median().sort_values().index
    sns.boxplot(data=df, x=col, y=TARGET, order=order, ax=ax)
    ax.set_title(f"Emission by {col}", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("Carbon Emission by Key Categories", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/emission_by_category.png", dpi=150)
plt.close()

# Correlations

print("\nCorrelations")

corr_with_target = df[numeric_cols.tolist() + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print(corr_with_target)

fig, ax = plt.subplots(figsize=(8, 5))
corr_with_target.plot(kind="barh", ax=ax, color=["#d62728" if v > 0 else "#1f77b4" for v in corr_with_target])
ax.set_xlabel("Pearson Correlation with CarbonEmission")
ax.set_title("Feature-Target Correlations")
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/feature_target_correlation.png", dpi=150)
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[numeric_cols.tolist() + [TARGET]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# Outliers

print("\nOutliers (IQR method)")

for col in numeric_cols.tolist() + [TARGET]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    if n_outliers > 0:
        print(f"  {col}: {n_outliers} outliers ({n_outliers / len(df) * 100:.1f}%)")

# Multi-select columns

print("\nRecycling & cooking")

df["Recycling_parsed"] = df["Recycling"].apply(ast.literal_eval)
df["Cooking_parsed"] = df["Cooking_With"].apply(ast.literal_eval)

print("\nRecycling frequencies:")
recycling_flat = [item for sublist in df["Recycling_parsed"] for item in sublist]
print(pd.Series(recycling_flat).value_counts())

print("\nCooking method frequencies:")
cooking_flat = [item for sublist in df["Cooking_parsed"] for item in sublist]
print(pd.Series(cooking_flat).value_counts())

print(f"\nAvg items recycled: {df['Recycling_parsed'].apply(len).mean():.2f}")
print(f"Avg cooking methods: {df['Cooking_parsed'].apply(len).mean():.2f}")

df = df.drop(columns=["Recycling_parsed", "Cooking_parsed"])
