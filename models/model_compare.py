import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from utils import apply_scaled_features

# Load engineered dataset.
df = pd.read_csv(os.path.join(BASE_DIR, "data", "carbon_engineered.csv"))
TARGET = "CarbonEmission"

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# Split train and split test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit consumption scaler on training data only
consumption_scaler = MinMaxScaler()
consumption_scaler.fit(X_train[["Monthly Grocery Bill", "How Many New Clothes Monthly"]])

X_train = apply_scaled_features(X_train, consumption_scaler)
X_test = apply_scaled_features(X_test, consumption_scaler)

# Save for the app
os.makedirs("models", exist_ok=True)
joblib.dump(consumption_scaler, "models/consumption_scaler.pkl")
print("Saved scaler to models/")

# Column groups for the preprocessor
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

categorical_columns = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
nominal_columns = [col for col in categorical_columns if col not in ordinal_columns]
numeric_columns = [col for col in X_train.columns if col not in categorical_columns]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

ordinal_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

nominal_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_columns),
    ("ord", ordinal_transformer, ordinal_columns),
    ("nom", nominal_transformer, nominal_columns)
])


def evaluate(name, pipeline, X_tr, y_tr, X_te, y_te, cv_score=None):
    """Evaluate a fitted model and return metrics dict."""
    pred = pipeline.predict(X_te)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    rmse = root_mean_squared_error(y_te, pred)

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


# Naive baseline - just predict the mean every time
# Gives context for how much the actual models improve over guessing
train_mean = y_train.mean()
baseline_pred = np.full(len(y_test), train_mean)
print("\nBASELINE (predict training mean)")
print(f"MAE:  {mean_absolute_error(y_test, baseline_pred):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, baseline_pred):.2f}")
print(f"R²:   {r2_score(y_test, baseline_pred):.4f}")


results = []
trained_models = {}

# 1. Linear Regression
print("\nTRAINING LINEAR REGRESSION")
linear_model = Pipeline([
    ("preprocessor", preprocessor),
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
    rf_pipeline, rf_param_dist,
    n_iter=50, scoring="r2", cv=5,
    verbose=1, random_state=42, n_jobs=-1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print("Best params:", rf_search.best_params_)

results.append(
    evaluate("Random Forest (Tuned)", best_rf,
             X_train, y_train, X_test, y_test,
             cv_score=rf_search.best_score_)
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
    gb_pipeline, gb_param_dist,
    n_iter=50, scoring="r2", cv=5,
    verbose=1, random_state=42, n_jobs=-1
)

gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_
print("Best params:", gb_search.best_params_)

results.append(
    evaluate("Gradient Boosting (Tuned)", best_gb,
             X_train, y_train, X_test, y_test,
             cv_score=gb_search.best_score_)
)
trained_models["Gradient Boosting (Tuned)"] = best_gb

# 4. XGBoost
print("\nTUNING XGBOOST")
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        random_state=42, n_jobs=-1,
        verbosity=0, tree_method="hist"
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
    xgb_pipeline, xgb_param_dist,
    n_iter=50, scoring="r2", cv=5,
    verbose=1, random_state=42, n_jobs=-1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print("Best params:", xgb_search.best_params_)

results.append(
    evaluate("XGBoost (Tuned)", best_xgb,
             X_train, y_train, X_test, y_test,
             cv_score=xgb_search.best_score_)
)
trained_models["XGBoost (Tuned)"] = best_xgb

# Summary
results_df = pd.DataFrame(results).sort_values("CV R² (5-fold)", ascending=False)

print("\nSUMMARY TABLE")
print(results_df.to_string(index=False))

best_row = results_df.iloc[0]
best_name = best_row["Model"]
best_model = trained_models[best_name]

print(f"\nBEST MODEL: {best_name}")
print(f"CV R²:  {best_row['CV R² (5-fold)']:.4f}")
print(f"Test R²: {best_row['R²']:.4f}")
print(f"MAE:    {best_row['MAE']:.2f}")
print(f"RMSE:   {best_row['RMSE']:.2f}")

joblib.dump(best_model, "models/best_model_new.pkl")
print("\nModel saved to: models/best_model_new.pkl")

results_df.to_csv("models/model_comparison_results.csv", index=False)
print("Results saved to: models/model_comparison_results.csv")

# Feature importance (only works for tree models)
tree_models = ["Random Forest (Tuned)", "Gradient Boosting (Tuned)", "XGBoost (Tuned)"]

if best_name in tree_models:
    try:
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        importances = best_model.named_steps["model"].feature_importances_

        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        print("\nTOP 15 FEATURES")
        print(feat_imp.head(15).to_string(index=False))

        feat_imp.to_csv("models/feature_importance.csv", index=False)

    except Exception as e:
        print(f"\nCouldn't get feature importance: {e}")

# Plots
pred_test = best_model.predict(X_test)
residuals = y_test - pred_test

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, pred_test, alpha=0.3, edgecolors="none")
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", lw=2, label="Perfect prediction"
)
axes[0].set_xlabel("Actual (kg CO₂e/month)")
axes[0].set_ylabel("Predicted (kg CO₂e/month)")
axes[0].set_title(f"Actual vs Predicted — {best_name}")
axes[0].legend()

axes[1].scatter(pred_test, residuals, alpha=0.3, edgecolors="none")
axes[1].axhline(0, color="r", linestyle="--", lw=2)
axes[1].set_xlabel("Predicted (kg CO₂e/month)")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("models/evaluation_plots.png", dpi=150)
plt.show()

# Learning curve
print("\nGenerating learning curve...")

X_full = df.drop(columns=[TARGET]).copy()
X_full = apply_scaled_features(X_full, consumption_scaler)

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_full, y,
    cv=5, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), label="Training R²")
ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation R²")
ax.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
ax.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
ax.set_xlabel("Training set size")
ax.set_ylabel("R²")
ax.set_title(f"Learning Curve — {best_name}")
ax.legend()
plt.tight_layout()
plt.savefig("models/learning_curve.png", dpi=150)
plt.show()

print("\nOrdinal columns:", ordinal_columns)
print("Nominal columns:", nominal_columns)
print("Numeric columns:", numeric_columns)
