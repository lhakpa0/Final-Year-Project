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
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


# Load engineered dataset
df = pd.read_csv("data/carbon_engineered.csv")
TARGET = "CarbonEmission"

# Separate features and target
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# Define ordinal columns and their order
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
nominal_columns = [col for col in categorical_columns if col not in ordinal_columns]
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

# Helper function to evaluate a fitted model
def evaluate(name, pipeline, X_tr, y_tr, X_te, y_te, cv_score=None):
    pred = pipeline.predict(X_te)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    rmse = root_mean_squared_error(y_te, pred)

    # If CV score is not already available, calculate it here
    if cv_score is None:
        cv_score = cross_val_score(
            pipeline, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1
        ).mean()

    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"CV R² (5-fold): {cv_score:.4f}")

    return {
        "Model": name,
        "R²": round(r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "CV R² (5-fold)": round(cv_score, 4)
    }

results = []
trained_models = {}

# 1. Linear Regression
print("\nTRAINING LINEAR REGRESSION")
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
results.append(
    evaluate("Linear Regression", linear_model, X_train, y_train, X_test, y_test)
)
trained_models["Linear Regression"] = linear_model

# 2. Random Forest
print("\nTUNING RANDOM FOREST")
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
    rf_pipeline,
    rf_param_dist,
    n_iter=10,
    scoring="r2",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print("Best params:", rf_search.best_params_)

results.append(
    evaluate(
        "Random Forest (Tuned)",
        best_rf,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_score=rf_search.best_score_
    )
)
trained_models["Random Forest (Tuned)"] = best_rf

# 3. Gradient Boosting
print("\nTUNING GRADIENT BOOSTING")
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
    gb_pipeline,
    gb_param_dist,
    n_iter=10,
    scoring="r2",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_
print("Best params:", gb_search.best_params_)

results.append(
    evaluate(
        "Gradient Boosting (Tuned)",
        best_gb,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_score=gb_search.best_score_
    )
)
trained_models["Gradient Boosting (Tuned)"] = best_gb

# 4. XGBoost
print("\nTUNING XGBOOST")
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist"
    ))
])

xgb_param_dist = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "model__reg_alpha": [0, 0.1, 0.5],
    "model__reg_lambda": [1, 1.5, 2]
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    xgb_param_dist,
    n_iter=10,
    scoring="r2",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print("Best params:", xgb_search.best_params_)

results.append(
    evaluate(
        "XGBoost (Tuned)",
        best_xgb,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_score=xgb_search.best_score_
    )
)
trained_models["XGBoost (Tuned)"] = best_xgb

# Summary table
results_df = pd.DataFrame(results).sort_values("CV R² (5-fold)", ascending=False)

print("\nSUMMARY TABLE")
print(results_df.to_string(index=False))

# Select and save best model
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

results_df.to_csv("models/model_comparison_results.csv", index=False)
print("Results table saved to: models/model_comparison_results.csv")

# Feature importance for tree-based models
tree_models = [
    "Random Forest (Tuned)",
    "Gradient Boosting (Tuned)",
    "XGBoost (Tuned)"
]

if best_name in tree_models:
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

print("\nOrdinal columns:", ordinal_columns)
print("Nominal columns:", nominal_columns)
print("Numeric columns:", numeric_columns)