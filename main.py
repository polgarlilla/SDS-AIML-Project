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

#Descriptive statistics
from descriptive_stats import summarize_features_and_target
summary_features, summary_target = summarize_features_and_target(X, y)
summary_features, summary_target

#Data preprocessing for modeling: 
#scaling numeric features and one-hot encoding categorical ones

from features import preprocessor_for_modeling
preprocess = preprocessor_for_modeling()

#Models
#Linear regression (baseline model)
import models_proba
model_baseline, y_pred_cv_baseline = models_proba.lin_reg_model(preprocess,X, y)

import evaluate
evaluate.evaluate_model(model_baseline, X, y)

#Lasso model
model_lasso, y_pred_cv_lasso = models_proba.lasso_model(preprocess, X, y)
evaluate.evaluate_model(model_lasso, X, y)