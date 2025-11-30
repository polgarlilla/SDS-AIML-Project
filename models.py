import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV

def lin_reg_model(preprocess, X, y):
    """
    Baseline logistic regression model using 5-fold cross-validation.
    """
    preprocess = preprocess
    # Logistic regression baseline (no hyperparameter tuning)
    log_reg = LogisticRegression(max_iter=100, random_state=3)

    # Pipeline: preprocess -> logistic regression
    model = make_pipeline(preprocess, log_reg)

    # 5-fold stratified CV for predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    # Out-of-fold predictions (each point predicted by a model
    # that did not see it during training)
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, method="predict")

    # Fit final model on all data so we can reuse it later
    model.fit(X, y)

    return model, y_pred_cv

def lasso_model(preprocess, X, y):
    """
    L1-regularized logistic regression (Lasso).
    Returns fitted model + CV predictions.
    """

    preprocess = preprocess

    log_reg_l1 = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=100,
        random_state=3
    )

    model = make_pipeline(preprocess, log_reg_l1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    y_pred_cv = cross_val_predict(model, X, y, cv=cv, method="predict")

    model.fit(X, y)

    return model, y_pred_cv

#KNN
def knn_model(preprocess, X, y, cv_splits=5):
    """
    KNN classifier with cosine distance and GridSearchCV over n_neighbors.

    """
    preprocess = preprocess

    # Base KNN model
    knn_clf = KNeighborsClassifier(
        weights="distance",
        metric="cosine"
    )

    pipe = make_pipeline(preprocess, knn_clf)

    # Hyperparameter grid
    param_grid_knn = {
        "kneighborsclassifier__n_neighbors": [3, 5, 10, 15, 25]
    }

    # Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=3)

    # Grid search over n_neighbors
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_knn,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    # Best model from the search
    best_model = grid_search.best_estimator_

    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, method="predict")

    # Fit best model on all data
    best_model.fit(X, y)

    return best_model, y_pred_cv, grid_search

# Decision Tree
def decision_tree_model(preprocess,X, y):
    """
    Decision tree classifier with GridSearchCV over:
      - max_depth
      - min_samples_leaf
    """

    preprocess = preprocess

    # Base decision tree classifier
    tree_clf = DecisionTreeClassifier(
        criterion="entropy",
        random_state=3
    )

    # Pipeline: preprocess -> decision tree
    pipe = make_pipeline(preprocess, tree_clf)

    # Parameter grid (pipeline step name = 'decisiontreeclassifier')
    param_grid_tree = {
        "decisiontreeclassifier__max_depth": [2, 3, 5, 10],
        "decisiontreeclassifier__min_samples_leaf": [1, 5, 10, 25],
    }

    # Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    # Grid search
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_tree,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    # Fit the grid search on the full dataset
    grid_search.fit(X, y)

    # Best pipeline
    best_model = grid_search.best_estimator_

    # Cross-validated predictions from the best model
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, method="predict")

    # Fit best model on *all* data
    best_model.fit(X, y)

    return best_model, y_pred_cv, grid_search