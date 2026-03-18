import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error

# Load engineered dataset
df = pd.read_csv("data/carbon_engineered.csv")
TARGET = "CarbonEmission"

# Separate input features and target variable
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# Define ordinal columns and their category order
ordinal_columns = [
    "How Often Shower",
    "Frequency of Traveling by Air",
    "Waste Bag Size",
    "Energy efficiency",
    "Social Activity"
]

ordinal_categories = [
    ["less frequently", "daily", "more frequently", "twice a day"],
    ["never", "rarely", "frequently", "very frequently"],
    ["small", "medium", "large", "extra large"],
    ["No", "Sometimes", "Yes"],
    ["never", "sometimes", "often"]
]

# Identify nominal and numeric columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
nominal_columns = [col for col in categorical_columns
                   if col not in ordinal_columns
                   and col not in ["Recycling", "Cooking_With"]]
numeric_columns = [col for col in X.columns if col not in categorical_columns]

# Preprocessing for numeric columns
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Preprocessing for ordinal columns
ordinal_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

# Preprocessing for nominal columns
nominal_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_columns),
    ("ord", ordinal_transformer, ordinal_columns),
    ("nom", nominal_transformer, nominal_columns)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline linear regression model
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),
    ("model", LinearRegression())
])

# Random Forest pipeline and search space
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

rf_param_dist = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None]
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_dist,
    n_iter=10,
    scoring="r2",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting pipeline and search space
gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

gb_param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__subsample": [0.8, 0.9, 1.0]
}

gb_search = RandomizedSearchCV(
    estimator=gb_pipeline,
    param_distributions=gb_param_dist,
    n_iter=10,
    scoring="r2",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train models and store results
results = []
trained_models = {}

print("\nTRAINING LINEAR REGRESSION")
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = root_mean_squared_error(y_test, linear_pred)
linear_cv = cross_val_score(
    linear_model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
).mean()

results.append({
    "Model": "Linear Regression",
    "R²": round(linear_r2, 4),
    "MAE": round(linear_mae, 2),
    "RMSE": round(linear_rmse, 2),
    "CV R² (5-fold)": round(linear_cv, 4)
})
trained_models["Linear Regression"] = linear_model

print(f"R²: {linear_r2:.4f}")
print(f"MAE: {linear_mae:.2f}")
print(f"RMSE: {linear_rmse:.2f}")
print(f"CV R² (5-fold): {linear_cv:.4f}")

print("\nTUNING RANDOM FOREST")
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = root_mean_squared_error(y_test, rf_pred)

results.append({
    "Model": "Random Forest (Tuned)",
    "R²": round(rf_r2, 4),
    "MAE": round(rf_mae, 2),
    "RMSE": round(rf_rmse, 2),
    "CV R² (5-fold)": round(rf_search.best_score_, 4)
})
trained_models["Random Forest (Tuned)"] = best_rf

print("Best parameters:", rf_search.best_params_)
print(f"R²: {rf_r2:.4f}")
print(f"MAE: {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
print(f"CV R² (5-fold): {rf_search.best_score_:.4f}")

print("\nTUNING GRADIENT BOOSTING")
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_
gb_pred = best_gb.predict(X_test)
gb_r2 = r2_score(y_test, gb_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = root_mean_squared_error(y_test, gb_pred)

results.append({
    "Model": "Gradient Boosting (Tuned)",
    "R²": round(gb_r2, 4),
    "MAE": round(gb_mae, 2),
    "RMSE": round(gb_rmse, 2),
    "CV R² (5-fold)": round(gb_search.best_score_, 4)
})
trained_models["Gradient Boosting (Tuned)"] = best_gb

print("Best parameters:", gb_search.best_params_)
print(f"R²: {gb_r2:.4f}")
print(f"MAE: {gb_mae:.2f}")
print(f"RMSE: {gb_rmse:.2f}")
print(f"CV R² (5-fold): {gb_search.best_score_:.4f}")

# Create summary table of model results
results_df = pd.DataFrame(results).sort_values("CV R² (5-fold)", ascending=False)

print("\nSUMMARY TABLE")
print(results_df.to_string(index=False))

# Select and save the best-performing model
best_row = results_df.iloc[0]
best_name = best_row["Model"]
best_model = trained_models[best_name]

print("\nBEST MODEL SELECTED")
print(f"Name           : {best_name}")
print(f"CV R²          : {best_row['CV R² (5-fold)']:.4f}")
print(f"Test R²        : {best_row['R²']:.4f}")
print(f"Test MAE       : {best_row['MAE']:.2f}")
print(f"Test RMSE      : {best_row['RMSE']:.2f}")

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model_new.pkl")
print("\nBest model saved to: models/best_model_new.pkl")

# Save model comparison results
results_df.to_csv("models/model_comparison_results.csv", index=False)
print("Results table saved to: models/model_comparison_results.csv")

# Extract feature importance for tree-based models
if best_name in ["Random Forest (Tuned)", "Gradient Boosting (Tuned)"]:
    try:
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        importances = best_model.named_steps["model"].feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        print("\nTOP 15 IMPORTANT FEATURES")
        print(feature_importance_df.head(15).to_string(index=False))

        feature_importance_df.to_csv("models/feature_importance.csv", index=False)
        print("\nFeature importance saved to: models/feature_importance.csv")

    except Exception as e:
        print("\nCould not extract feature importance.")
        print("Reason:", e)

# Print column groupings for checking
print("Ordinal columns:", ordinal_columns)
print("Nominal columns:", nominal_columns)
print("Numeric columns:", numeric_columns)