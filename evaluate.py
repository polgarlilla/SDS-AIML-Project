#Evaluation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
)


def evaluate_model(model, X, y):
    """
    Evaluates a model:
    - 5-fold cross-validated AUC, accuracy, precision, recall
    - Confusion matrix and ROC curve from an 80/20 train-test split.
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring_metrics = {
        "roc_auc": "AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
    }

    metrics_cv = {}

    print("\n=== 5-fold cross-validated metrics ===")
    for scoring, nice_name in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        mean_score = scores.mean()
        std_score = scores.std()
        metrics_cv[nice_name.lower()] = {"mean": mean_score, "std": std_score}
        print(f"{nice_name} (CV mean ± std): {mean_score:.3f} ± {std_score:.3f}")

    # 80/20 train-test split for visualizations
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    # Predictions and probabilities on test set
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Confusion matrix
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    n = cm.shape[0]
    correct_mask = np.eye(n)
    cmap = ListedColormap(['#E53935', '#43A047'])  # red for errors, green for correct

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(correct_mask, cmap=cmap, vmin=0, vmax=1)

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    for (i, j), val in np.ndenumerate(cm):
        txt = f"{val}\n({cm_norm[i, j]:.2f})"
        ax.text(j, i, txt, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Pred {l}" for l in labels])
    ax.set_yticklabels([f"True {l}" for l in labels])
    ax.set_title("Confusion matrix (80/20 split)")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    plt.tight_layout()
    plt.show()

    # ROC curve + AUC
    auc_test = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"ROC curve (AUC on test = {auc_test:.3f})")
    plt.show()

    return metrics_cv
