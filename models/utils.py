AIR_MAP = {
    "never": 0,
    "rarely": 1,
    "frequently": 2,
    "very frequently": 3
}

WASTE_SIZE_MAP = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "extra large": 4
}

# features to hide from the SHAP chart (not actionable for the user)
DEMOGRAPHIC_FEATURES = {"Sex", "Body Type"}


def apply_basic_features(df):
# Create a waste score by multiplying bag size by weekly count.
    df = df.copy()

    df["WasteSizeScore"] = df["Waste Bag Size"].map(WASTE_SIZE_MAP)
    df["WasteScore"] = df["Waste Bag Weekly Count"] * df["WasteSizeScore"]
    df = df.drop(columns=["WasteSizeScore"])
    return df

def apply_scaled_features(df, consumption_scaler):
    df = df.copy()
    scaled = consumption_scaler.transform(
        df[["Monthly Grocery Bill", "How Many New Clothes Monthly"]]
    )
    df["ConsumptionScore"] = scaled[:, 0] + scaled[:, 1]
    return df
