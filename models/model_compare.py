import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load engineered dataset
df = pd.read_csv("data/carbon_engineered.csv")
TARGET = "CarbonEmission"

# Split features/target
X = df.drop(columns=[TARGET]).copy()  
y = df[TARGET]



# Ordinal encoding
ordinal_mappings = {
    "How Often Shower": {
        "less frequently": 0,
        "daily": 1,
        "more frequently": 2,
        "twice a day": 3
    },
    "Frequency of Traveling by Air": {
        "never": 0,
        "rarely": 1,
        "frequently": 2,
        "very frequently": 3
    },
    "Waste Bag Size": {
        "small": 0,
        "medium": 1,
        "large": 2,
        "extra large": 3
    },
    "Energy efficiency": {
        "No": 0,
        "Sometimes": 1,
        "Yes": 2
    }
}

# warn if any NaNs appear after encoding
for col, mapping in ordinal_mappings.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)
        nan_count = X[col].isna().sum()
        if nan_count > 0:
            print(f"WARNING: '{col}' has {nan_count} unmapped value(s) → NaN. Check your CSV for unexpected categories.")

# One-hot encoding for remaining categorical columns
X = pd.get_dummies(X, drop_first=True)
# remove any remaining missing values
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    )
}

# Store results in a DataFrame for better visualization
results = []

print("\n MODEL COMPARISON \n")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2  = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    # Cross-validation for more reliable estimates
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()

    results.append({"Model": name, "R²": round(r2, 4), "MAE": round(mae, 2), "CV R² (5-fold)": round(cv_r2, 4)})

    print(f"{name}")
    print(f"  R²           : {r2:.4f}")
    print(f"  MAE          : {mae:.2f}")
    print(f"  CV R² (5-fold): {cv_r2:.4f}")
    print("-" * 32)

# Summary table
results_df = pd.DataFrame(results).sort_values("CV R² (5-fold)", ascending=False)
print("\n=== SUMMARY TABLE ===")
print(results_df.to_string(index=False))

best_model = models["Gradient Boosting"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CarbonEmission")
plt.ylabel("Predicted CarbonEmission")
plt.title("Predicted vs Actual (Gradient Boosting)")
# ideal line y = x
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx])
plt.show()

# residuals
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted CarbonEmission")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted (Gradient Boosting)")
plt.show()

gb = models["Gradient Boosting"]
gb.fit(X_train, y_train)

importances = pd.Series(gb.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure()
top_features.sort_values().plot(kind="barh")
plt.xlabel("Importance Score")
plt.title("Top 15 Feature Importances (Gradient Boosting)")
plt.show()



# Save trained model
joblib.dump(best_model, "models/gb_model.pkl")

# Save feature column order
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model saved successfully.")