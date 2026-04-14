import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, learning_curve

def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test, cv_score=None):
    # Calculate R2, MAE, RMSE and cross validation score for a trained model
    pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    if cv_score is None:
        cv_score = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
        ).mean()

    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"CV R2 (5-fold): {cv_score:.4f}")

    return {
        "Model": name,
        "R2": round(r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "CV R2 (5-fold)": round(cv_score, 4),
    }

def print_baseline(y_train, y_test):
    # Show how a simple mean predictor performs as a comparison baseline
    train_mean = y_train.mean()
    baseline_pred = np.full(len(y_test), train_mean)
    print("\nBASELINE (predict training mean)")
    print(f"MAE:  {mean_absolute_error(y_test, baseline_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, baseline_pred):.2f}")
    print(f"R2:   {r2_score(y_test, baseline_pred):.4f}")

def save_feature_importance(model, save_path):
    # Extract and save feature importances from tree based models
    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_

        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)

        print("\nTOP 15 FEATURES")
        print(feat_imp.head(15).to_string(index=False))

        feat_imp.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")
    except Exception as e:
        print(f"\nCouldn't get feature importance: {e}")

def plot_evaluation(best_model, best_name, X_test, y_test, save_path):
    # Plot actual vs predicted and residual charts
    pred_test = best_model.predict(X_test)
    residuals = y_test - pred_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, pred_test, alpha=0.3, edgecolors="none")
    axes[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--", lw=2, label="Perfect prediction",
    )
    axes[0].set_xlabel("Actual (kg CO2e/month)")
    axes[0].set_ylabel("Predicted (kg CO2e/month)")
    axes[0].set_title(f"Actual vs Predicted - {best_name}")
    axes[0].legend()

    axes[1].scatter(pred_test, residuals, alpha=0.3, edgecolors="none")
    axes[1].axhline(0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted (kg CO2e/month)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")

def plot_learning_curve(model, name, X, y, save_path):
    # Show how model performance changes with more training data
    print("\nGenerating learning curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Training R2")
    ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation R2")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.1,
    )
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.1,
    )
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R2")
    ax.set_title(f"Learning Curve - {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")

def plot_model_comparison(trained_models, X, y, save_path):
    # Compare all models on the same learning curve chart
    print("\nGenerating model comparison chart...")

    model_colors = {
        "Gradient Boosting": "#d62728",
        "XGBoost": "#1f77b4",
        "Linear Regression": "#2ca02c",
        "Random Forest": "#ff7f0e",
        "SVR": "#9467bd",
    }

    comparison_sizes = np.linspace(0.1, 1.0, 10)
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, model in trained_models.items():
        print(f"  Computing learning curve for {name}...")

        sizes, _, val_scores = learning_curve(
            model, X, y,
            train_sizes=comparison_sizes,
            cv=5, scoring="r2",
            n_jobs=-1,
        )
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        color = model_colors.get(name, "#333333")
        ax.plot(sizes, val_mean, label=name, color=color, linewidth=2)
        ax.fill_between(
            sizes, val_mean - val_std, val_mean + val_std,
            alpha=0.15, color=color,
        )
    ax.set_xlabel("Training set size", fontsize=13)
    ax.set_ylabel("R2 score", fontsize=13)
    ax.set_title("Model Comparison - CV R2 vs Training Size\n(Mean +/- Std across 5 folds)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")
