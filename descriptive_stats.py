import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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


def multicollinearity_corr_heatmap(df):

  """
  Computes the correlation matrix for all numeric columns
  in the DataFrame and plots a correlation heatmap.
  """
  corr = df.corr()
  plt.figure(figsize=(12, 8))
  sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
  plt.title("Correlation Heatmap")
  plt.show()

    
def multicollinearity_vif(df):

  """
  Computes the VIF for all numeric columns
  """
  X = df.select_dtypes(include=["float64", "int64", "int8"]).dropna()

  X_const = sm.add_constant(X)

  vif_df = pd.DataFrame()
  vif_df["feature"] = X.columns
  vif_df["VIF"] = [variance_inflation_factor(X_const.values, i+1)
                  for i in range(len(X.columns))]

  print(vif_df)
  return vif_df