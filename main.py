'''
This script contains the main workflow for data loading, preprocessing, modeling, and evaluation.
Random state is set to 3 everywhere.
'''

#Data loading and preprocessing

from sklearn.linear_model import LogisticRegression
from data import (
    read_and_inspect_data,
    create_holdout_set,
    cols_to_keep,
    qualification_map,
    categoricals,
    preprocess_data,
)

df = read_and_inspect_data("data/Dropout_Data.xlsx", "Original Data")
df,df_holdout=create_holdout_set(df)
X, y = preprocess_data(df, cols_to_keep, qualification_map, categoricals)

######################################################################################################

#Descriptive statistics
from descriptive_stats import (
    plot_dummy_histograms,
    summarize_features_and_target,
    plot_target_distribution,
    plot_numeric_histogram,
    plot_boxplot_two_features,
    plot_parental_qualification_distribution,
    dummy_cols,
    plot_dummy_histograms,
)

summary_features, summary_target = summarize_features_and_target(X, y)
summary_features, summary_target

plot_target_distribution(y)
plot_numeric_histogram(X,column="Previous qualification (grade)")
plot_boxplot_two_features(X, "Inflation rate", "GDP")
plot_parental_qualification_distribution(X,mother_col="Mother's qualification",
    father_col="Father's qualification",levels=(0, 1, 2, 3, 4),)
plot_dummy_histograms(X, dummy_cols)



######################################################################################################

#Data preprocessing for modeling: 
#scaling numeric features and one-hot encoding categorical ones

from features import preprocessor_for_modeling
preprocess = preprocessor_for_modeling()

######################################################################################################

# Models
import models
from evaluate import evaluate_model, evaluation_on_holdout
#We used 80-20 train-test splits to visualize confusion matrices and  ROC curves

# Linear regression (baseline model)
model_baseline, y_pred_cv_baseline = models.lin_reg_model(preprocess, X, y)
evaluate_model(model_baseline, X, y, model_name="Baseline Logistic Regression")

# Lasso model
model_lasso, y_pred_cv_lasso = models.lasso_model(preprocess, X, y)
evaluate_model(model_lasso, X, y, model_name="Lasso Logistic Regression")

# KNN
model_knn, y_pred_cv_knn, grid_search_knn = models.knn_model(preprocess, X, y)
evaluate_model(model_knn, X, y, model_name="KNN Classifier")

# Decision Tree
model_tree, y_pred_cv_tree, grid_search_tree = models.decision_tree_model(preprocess, X, y)
evaluate_model(model_tree, X, y, model_name="Decision Tree Classifier")

######################################################################################################

#Checking multicollinearity
from descriptive_stats import multicollinearity_corr_heatmap
multicollinearity_corr_heatmap(X)

from descriptive_stats import multicollinearity_vif
multicollinearity_vif(X)
#We can see that there is no significant multicollinearity among the features.

######################################################################################################

#Final evaluation on holdout set
X_holdout, y_holdout = preprocess_data(df_holdout, cols_to_keep, qualification_map, categoricals)

print("\n=== BASELINE MODEL (Holdout) ===")
evaluation_on_holdout(model_baseline, X_holdout, y_holdout,model_name="Baseline Logistic Regression")

print("\n=== LASSO MODEL (Holdout) ===")
evaluation_on_holdout(model_lasso, X_holdout, y_holdout,model_name="Lasso Logistic Regression")

print("\n=== KNN MODEL (Holdout) ===")
evaluation_on_holdout(model_knn, X_holdout, y_holdout,model_name="KNN Classifier")

print("\n=== DECISION TREE MODEL (Holdout) ===")
evaluation_on_holdout(model_tree, X_holdout, y_holdout,model_name="Decision Tree Classifier")
#based on accuracy, the decision tree performs best on the holdout set