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
from descriptive_stats import summarize_features_and_target
summary_features, summary_target = summarize_features_and_target(X, y)
summary_features, summary_target

######################################################################################################

#Data preprocessing for modeling: 
#scaling numeric features and one-hot encoding categorical ones

from features import preprocessor_for_modeling
preprocess = preprocessor_for_modeling()

######################################################################################################

#Models
#Linear regression (baseline model)
import models
model_baseline, y_pred_cv_baseline = models.lin_reg_model(preprocess,X, y)

import evaluate
evaluate.evaluate_model(model_baseline, X, y)

#Lasso model
model_lasso, y_pred_cv_lasso = models.lasso_model(preprocess, X, y)
evaluate.evaluate_model(model_lasso, X, y)

#KNN
model_knn, y_pred_cv_knn, grid_search_knn = models.knn_model(preprocess, X, y)
evaluate.evaluate_model(model_knn, X, y)

#Decision Tree
model_tree, y_pred_cv_tree, grid_search_tree = models.decision_tree_model(preprocess, X, y)
evaluate.evaluate_model(model_tree, X, y)

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

from evaluation import evaluation_on_holdout

print("\n=== BASELINE MODEL (Holdout) ===")
evaluation_on_holdout(model_baseline, X_holdout, y_holdout)

print("\n=== LASSO MODEL (Holdout) ===")
evaluation_on_holdout(model_lasso, X_holdout, y_holdout)

print("\n=== KNN MODEL (Holdout) ===")
evaluation_on_holdout(model_knn, X_holdout, y_holdout)

print("\n=== DECISION TREE MODEL (Holdout) ===")
evaluation_on_holdout(model_tree, X_holdout, y_holdout)
#based on accuracy, the decision tree performs best on the holdout set