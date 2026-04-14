# Re-export shared functions and constants so app.py can import from utils
from features import apply_basic_features, apply_scaled_features, DEMOGRAPHIC_FEATURES

__all__ = ["apply_basic_features", "apply_scaled_features", "DEMOGRAPHIC_FEATURES"]
