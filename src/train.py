from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from evaluate import (
    evaluate_model,
    plot_evaluation,
    plot_learning_curve,
    plot_model_comparison,
    print_baseline,
    save_feature_importance,
)
from features import PROCESSED_DATA_PATH, apply_scaled_features
from preprocessing import build_preprocessor
from models import ALL_MODELS

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

def train_single_model(model_class, preprocessor, X_train, y_train, X_test, y_test):
    # Train one model with optional hyperparameter tuning
    name = model_class.MODEL_NAME
    pipeline = model_class.get_pipeline(preprocessor)

    if model_class.PARAM_DIST is not None:
        print(f"\nTUNING {name.upper()}")
        search = RandomizedSearchCV(
            pipeline,
            model_class.PARAM_DIST,
            n_iter=50,
            scoring="r2",
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        cv_score = search.best_score_
        print("Best params:", search.best_params_)
    else:
        print(f"\nTRAINING {name.upper()}")
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        cv_score = None

    metrics = evaluate_model(name, best_pipeline, X_train, y_train, X_test, y_test, cv_score)
    return name, best_pipeline, metrics


def main():
    # Load processed data and split into train/test
    df = pd.read_csv(PROCESSED_DATA_PATH)
    target = "CarbonEmission"

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit scaler on training data only to avoid data leakage
    consumption_scaler = MinMaxScaler()
    consumption_scaler.fit(X_train[["Monthly Grocery Bill", "How Many New Clothes Monthly"]])

    X_train = apply_scaled_features(X_train, consumption_scaler)
    X_test = apply_scaled_features(X_test, consumption_scaler)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    joblib.dump(consumption_scaler, MODELS_DIR / "consumption_scaler.pkl")
    print("Saved scaler to models/")

    preprocessor, ordinal_cols, nominal_cols, numeric_cols = build_preprocessor(X_train)

    print_baseline(y_train, y_test)

    # Train all models and collect results
    results = []
    trained_models = {}

    for mod_class in ALL_MODELS:
        name, best_pipeline, metrics = train_single_model(
            mod_class, preprocessor, X_train, y_train, X_test, y_test
        )
        results.append(metrics)
        trained_models[name] = best_pipeline
        joblib.dump(best_pipeline, MODELS_DIR / mod_class.MODEL_FILE)

    # Pick the model with the highest cross validation R2
    results_df = pd.DataFrame(results).sort_values("CV R2 (5-fold)", ascending=False)
    print("\nSUMMARY TABLE")
    print(results_df.to_string(index=False))

    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    print(f"\nBEST MODEL: {best_name}")
    print(f"CV R2:  {best_row['CV R2 (5-fold)']:.4f}")
    print(f"Test R2: {best_row['R2']:.4f}")
    print(f"MAE:    {best_row['MAE']:.2f}")
    print(f"RMSE:   {best_row['RMSE']:.2f}")

    # Save the best model separately for the web app to load
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    print("\nModel saved to: models/best_model.pkl")

    results_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("Results saved to: results/metrics.csv")

    tree_models = {"Random Forest", "Gradient Boosting", "XGBoost"}
    if best_name in tree_models:
        save_feature_importance(best_model, RESULTS_DIR / "feature_importance.csv")

    X_full = df.drop(columns=[target]).copy()
    X_full = apply_scaled_features(X_full, consumption_scaler)

    plot_evaluation(best_model, best_name, X_test, y_test, PLOTS_DIR / "evaluation_plots.png")
    plot_learning_curve(best_model, best_name, X_full, y, PLOTS_DIR / "learning_curve.png")
    plot_model_comparison(trained_models, X_full, y, PLOTS_DIR / "model_comparison.png")

    print("\nOrdinal columns:", ordinal_cols)
    print("Nominal columns:", nominal_cols)
    print("Numeric columns:", numeric_cols)


if __name__ == "__main__":
    main()
