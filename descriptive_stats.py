import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import os

def summarize_features_and_target(X, y, output_dir="outputs"):
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

    summary_features.to_csv(os.path.join(output_dir, "summary_features.csv"))
    summary_target.to_csv(os.path.join(output_dir, "summary_target.csv"))

    return summary_features, summary_target


def plot_target_distribution(y,title="Dropout vs Graduate proportion (Target)",output_dir="outputs"):
    """
    Plots the class proportion of the target variable.
    """
    target_ratio = y.value_counts(normalize=True) * 100

    plt.figure()
    target_ratio.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Percentage (%)")
    plt.xlabel("Target class")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "target_distribution.png"),
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_numeric_histogram(X,column,bins=30,title=None,xlabel=None,):
    """
    Plots a histogram

    """
    if column not in X.columns:
        raise ValueError(f"Column '{column}' not found in X.")

    data = X[column].dropna()

    if title is None:
        title = f"Histogram of {column}"
    if xlabel is None:
        xlabel = column

    plt.figure()
    data.hist(bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()



def plot_boxplot_two_features(X,col1,col2,title=None,ylabel="Value"):
    """
    Plot side-by-side boxplots
    """
    if col1 not in X.columns:
        raise ValueError(f"Column '{col1}' not found in X.")
    if col2 not in X.columns:
        raise ValueError(f"Column '{col2}' not found in X.")

    if title is None:
        title = f"Boxplot of {col1} and {col2}"

    plt.figure()
    plt.boxplot(
        [X[col1].dropna(), X[col2].dropna()],
        labels=[col1, col2]
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_parental_qualification_distribution(X,
    mother_col="Mother's qualification",
    father_col="Father's qualification",
    levels=(0, 1, 2, 3, 4),title="Distribution of parental qualification levels"):
    """
    Plot grouped bar chart of parental qualification level proportions.

    """
    for col in [mother_col, father_col]:
        if col not in X.columns:
            raise ValueError(f"Column '{col}' not found in X.")

    mother_prop = X[mother_col].value_counts(normalize=True).sort_index()
    father_prop = X[father_col].value_counts(normalize=True).sort_index()

    qual_prop = pd.DataFrame({
        "Mother": mother_prop,
        "Father": father_prop
    })

    qual_prop = qual_prop.reindex(list(levels))

    ax = qual_prop.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel("Qualification level")
    ax.set_ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

dummy_cols = [
    "International",
    "Scholarship holder",
    "Gender",
    "Tuition fees up to date",
    "Debtor",
]



def plot_dummy_histograms(X,dummy_cols):
    """
    Plot histograms for multiple 0/1 dummy variables.
    """
    for col in dummy_cols:
        if col not in X.columns:
            raise ValueError(f"Dummy column '{col}' not found in X.")

        plt.figure()
        data = X[col].dropna()
        data = data[data.isin([0, 1])]

        plt.hist(data, bins=[-0.5, 0.5, 1.5])
        plt.xticks([0, 1], ["0", "1"])
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()




def multicollinearity_corr_heatmap(df,output_dir="outputs"):

  """
  Computes the correlation matrix for all numeric columns
  in the DataFrame and plots a correlation heatmap.
  """
  corr = df.corr()
  plt.figure(figsize=(12, 8))
  sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
  plt.title("Correlation Heatmap")
  fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"),
                dpi=300, bbox_inches="tight")
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