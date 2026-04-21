from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Linear Regression - the simplest model, used as a baseline for comparison with more complex models.
class LinearRegressionModel:
    MODEL_NAME = "Linear Regression"
    MODEL_FILE = "linear_regression.pkl"
    PARAM_DIST = None  # No hyperparameters to tune

    @staticmethod
    def get_pipeline(preprocessor):
        # Preprocessor is included in the pipeline to ensure that all data transformations are applied consistently during training and prediction.
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
        "model__min_samples_split": [2, 5, 10], # min samples needed to split a node
        "model__min_samples_leaf": [1, 2, 4], # min samples needed at a leaf node
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
        "model__n_estimators": [100, 200, 300], # number of boosting stages to perform
        "model__learning_rate": [0.01, 0.05, 0.1], # step size shrinkage used in update to prevent overfitting
        "model__max_depth": [2, 3, 4, 5], # maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
        "model__min_samples_split": [2, 5, 10], # min samples needed to split a node
        "model__min_samples_leaf": [1, 2, 4], # min samples needed at a leaf node
        "model__subsample": [0.8, 0.9, 1.0] # fraction of samples used for fitting the individual base learners
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
        "model__max_depth": [3, 4, 5, 6, 8], # maximum depth of a tree
        "model__subsample": [0.7, 0.8, 0.9, 1.0], # fraction of samples used for fitting the individual base learners
        "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0], # fraction of features used for fitting the individual base learners
        "model__reg_alpha": [0, 0.1, 0.5], # L1 regularization term on weights
        "model__reg_lambda": [1, 1.5, 2] # L2 regularization term on weights    
    }

    @staticmethod
    def get_pipeline(preprocessor):
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                objective="reg:squarederror", # regression with squared error loss function
                random_state=42,
                n_jobs=-1,
                verbosity=0,  # suppresses the training logs for cleaner output
                tree_method="hist" # use histogram-based algorithm for faster training on large datasets
            ))
        ])


# All four models used for training and comparison
ALL_MODELS = [
    LinearRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
]
