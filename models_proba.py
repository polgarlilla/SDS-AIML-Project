import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def lin_reg_model(preprocess, X, y):
    """
    Baseline logistic regression model using 5-fold cross-validation.
    """
    preprocess = preprocess
    # Logistic regression baseline (no hyperparameter tuning)
    log_reg = LogisticRegression(max_iter=100, random_state=42)

    # Pipeline: preprocess -> logistic regression
    model = make_pipeline(preprocess, log_reg)

    # 5-fold stratified CV for predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred_cv = cross_val_predict(model, X, y, cv=cv, method="predict")

    model.fit(X, y)

    return model, y_pred_cv
