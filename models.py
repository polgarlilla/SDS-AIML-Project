'''
This script contains functions for the models:
'''

# Importing packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV

# import .py files
from features import preprocessor_for_modeling
from evaluate import cv

# Baselin model: logistic regression
log_reg = LogisticRegression(max_iter=100, random_state=42)

baseline_model = make_pipeline(
    preprocessor_for_modeling,
    log_reg
)


# Lasso model
lasso_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    max_iter=100,
    random_state=3
)

lasso = make_pipeline(
    preprocessor_for_modeling,
    lasso_model
)


# KNN model
knn_model = KNeighborsClassifier(weights="distance",metric="cosine")

knn = make_pipeline(
    preprocessor_for_modeling,
    knn_model
)

param_grid_knn = {
  "kneighborsclassifier__n_neighbors": [3, 5, 10, 15, 25]}

grid_knn = GridSearchCV(
    estimator=knn,
    param_grid=param_grid_knn,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)


# Decision Tree model
tree_model = DecisionTreeClassifier(criterion="entropy")

tree = make_pipeline(
    preprocessor_for_modeling,
    tree_model
    )

param_grid_tree = {
    'model__max_depth': [2, 3, 5, 10],
    'model__min_samples_leaf': [1, 5, 10, 25]}

grid_tree = GridSearchCV(
    estimator = tree,
    param_grid = param_grid_tree,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1)
