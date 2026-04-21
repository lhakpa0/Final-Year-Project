# Carbon Emission Predictor

An explainable machine learning application that estimates an individual's monthly carbon footprint from lifestyle survey responses and highlights which behaviours are driving the result.

Individual carbon footprint estimation has traditionally relied on rule-based calculators that apply static emission factors to survey responses. While widely accessible, these tools do not capture the nonlinear interactions between lifestyle variables that drive emissions, nor do they explain which specific behaviours are most responsible for a given result. This project investigates whether supervised machine learning, combined with SHAP-based explainability, can serve as a complementary methodology for individual carbon footprint prediction from lifestyle survey data.

The project uses the *Individual Carbon Footprint Tracker* dataset, a synthetic dataset of 10,000 records available on Kaggle, containing 20 lifestyle features covering transport, diet, residential energy use, consumption habits, and waste generation. The target variable is monthly carbon emissions measured in kilograms of carbon dioxide equivalent. Four regression algorithms were trained and compared: Linear Regression, Random Forest, Gradient Boosting, and XGBoost. All models were evaluated using five-fold cross-validation on four metrics: R², mean absolute error, root mean squared error, and cross-validated R². XGBoost was selected as the best-performing model, achieving a test R² of 0.9865 and a mean absolute error of 87.91 kg CO₂e per month, representing approximately 3.9% of the dataset mean. SHAP (SHapley Additive exPlanations) values are computed at prediction time using TreeExplainer to provide per-prediction explanations. Demographic features such as sex and body type are deliberately excluded from user-facing explanations, as they are non-actionable. The system is deployed as a single-page Streamlit web application that accepts lifestyle survey inputs, generates a monthly carbon emission prediction, and displays a personalised SHAP bar chart alongside actionable reduction tips.

The principal limitation of the project is the synthetic nature of the training data, which prevents any claim of real-world predictive accuracy. The high-performance metrics are a characteristic of the data-generating process rather than evidence of genuine generalisation to real household surveys. The project is therefore framed as a proof of concept demonstrating the technical feasibility of combining supervised learning with SHAP-based explainability for individual carbon footprint estimation, rather than as a production-ready system.

## Workflow

1. Run exploratory analysis on the raw dataset.
2. Build engineered features from the raw data.
3. Train and compare multiple regression models.
4. Save trained artifacts and evaluation outputs.
5. Launch the Streamlit app for interactive predictions.

## Setup

```bash
pip install -r requirements.txt
python src/EDA_analysis.py
python src/features.py
python src/train.py
streamlit run app.py
```

## Project structure

├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/                   # Original dataset (Kaggle CSV)
│   └── processed/             # Engineered dataset produced by features.py
├── models/                    # Trained model files and preprocessing artifacts (.pkl)
├── notebooks/                 # Exploratory Jupyter notebooks
├── results/
│   ├── metrics.csv            # Model comparison metrics
│   ├── feature_importance.csv # Feature importances from best model
│   └── plots/                 # Evaluation, learning-curve, comparison, and EDA plots
└── src/
    ├── EDA_analysis.py        # Exploratory data analysis script
    ├── features.py            # Feature engineering pipeline
    ├── preprocessing.py       # ColumnTransformer for numeric/ordinal/nominal features
    ├── models.py              # Model class definitions and hyperparameter grids
    ├── train.py               # Model training, tuning, and artifact saving
    ├── evaluate.py            # Metrics and evaluation plots
    ├── predict.py             # Standalone prediction smoke test
    └── utils.py               # Shared helpers re-exported for app.py


## Important outputs
- `data/processed/carbon_engineered.csv` — engineered training dataset
- `models/*.pkl` — saved trained models and preprocessing artifacts
- `results/metrics.csv` — model comparison metrics
- `results/plots/` — evaluation, learning-curve, comparison, and EDA plots
