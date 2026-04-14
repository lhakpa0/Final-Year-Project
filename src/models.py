from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# Linear Regression
class LinearRegressionModel:
    MODEL_NAME = "Linear Regression"
    MODEL_FILE = "linear_regression.pkl"
    PARAM_DIST = None  # No hyperparameters to tune

    @staticmethod
    def get_pipeline(preprocessor):
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])


# Random Forest
class RandomForestModel:
    MODEL_NAME = "Random Forest"
    MODEL_FILE = "random_forest.pkl"
    PARAM_DIST = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    }

    @staticmethod
    def get_pipeline(preprocessor):
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
        ])


# Gradient Boosting
class GradientBoostingModel:
    MODEL_NAME = "Gradient Boosting"
    MODEL_FILE = "gradient_boosting.pkl"
    PARAM_DIST = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.8, 0.9, 1.0]
    }

    @staticmethod
    def get_pipeline(preprocessor):
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42))
        ])


# XGBoost
class XGBoostModel:
    MODEL_NAME = "XGBoost"
    MODEL_FILE = "xgboost_model.pkl"
    PARAM_DIST = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "model__reg_alpha": [0, 0.1, 0.5],
        "model__reg_lambda": [1, 1.5, 2]
    }

    @staticmethod
    def get_pipeline(preprocessor):
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist"
            ))
        ])


# All four models used for training and comparison
ALL_MODELS = [
    LinearRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
]
