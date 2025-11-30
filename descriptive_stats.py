import pandas as pd

def summarize_features_and_target(X, y):
    """
    Generates descriptive statistics tables for features (X) and target (y).
    Returns two DataFrames: summary of features,ó and summary of target
    """

    # Descriptive statistics for X

    desc_feat = X.describe(include="all").T

    desc_feat["dtype"] = X.dtypes.astype(str)
    desc_feat["nunique"] = X.nunique()
    desc_feat["missing_count"] = X.isna().sum()

    first_cols = ["dtype", "nunique", "missing_count"]
    other_cols = [c for c in desc_feat.columns if c not in first_cols]
    summary_features = desc_feat[first_cols + other_cols].copy()

    num_cols = summary_features.select_dtypes(include="number").columns
    summary_features[num_cols] = summary_features[num_cols].round(3)

    # Descriptive statistics for y

    desc_tgt = y.describe(include="all").to_frame().T

    desc_tgt["dtype"] = y.dtype
    desc_tgt["nunique"] = y.nunique()
    desc_tgt["missing_count"] = y.isna().sum()

    first_cols_t = ["dtype", "nunique", "missing_count"]
    other_cols_t = [c for c in desc_tgt.columns if c not in first_cols_t]
    summary_target = desc_tgt[first_cols_t + other_cols_t].copy()

    num_cols_t = summary_target.select_dtypes(include="number").columns
    summary_target[num_cols_t] = summary_target[num_cols_t].round(3)

    return summary_features, summary_target