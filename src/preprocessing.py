from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Ordered categories for ordinal encoding
ORDINAL_COLUMNS = [
    "How Often Shower",
    "Frequency of Traveling by Air",
    "Waste Bag Size",
    "Energy efficiency",
    "Social Activity",
]
ORDINAL_CATEGORIES = [
    ["less frequently", "daily", "more frequently", "twice a day"],
    ["never", "rarely", "frequently", "very frequently"],
    ["small", "medium", "large", "extra large"],
    ["No", "Sometimes", "Yes"],
    ["never", "sometimes", "often"],
]

def build_preprocessor(X_train):
    # Separate columns by type and build a transformer for each
    categorical_columns = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    ordinal_columns = [col for col in ORDINAL_COLUMNS if col in X_train.columns]
    nominal_columns = [col for col in categorical_columns if col not in ordinal_columns]
    numeric_columns = [col for col in X_train.columns if col not in categorical_columns]

    # Median fill for numbers
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    # Encode ordered categories as integers
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    # One hot encode unordered categories
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_columns),
        ("ord", ordinal_transformer, ordinal_columns),
        ("nom", nominal_transformer, nominal_columns),
    ])
    return preprocessor, ordinal_columns, nominal_columns, numeric_columns
