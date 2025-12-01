#Evaluation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import json
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score)

def _safe_model_name(model_name: str) -> str:
    """Convert model name to a safe file-name fragment."""
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

def evaluate_model(model, X, y, model_name="Model", output_dir="outputs"):
    """
    Evaluates a model:
    - 5-fold cross-validated AUC, accuracy, precision, recall
    - Confusion matrix and ROC curve from an 80/20 train-test split.
    - Saves outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = _safe_model_name(model_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    scoring_metrics = {
        "roc_auc": "AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
    }

    metrics_cv = {}

    print(f"\n=== 5-fold cross-validated metrics for {model_name} ===")
    for scoring, nice_name in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        mean_score = scores.mean()
        std_score = scores.std()
        metrics_cv[nice_name.lower()] = {
            "mean": float(mean_score),
            "std": float(std_score),
        }
        print(f"{nice_name} (CV mean ± std): {mean_score:.3f} ± {std_score:.3f}")

    # save CV metrics
    metrics_json_path = os.path.join(output_dir, f"{safe_name}_cv_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_cv, f, indent=4)

    metrics_df = pd.DataFrame(metrics_cv).T
    metrics_csv_path = os.path.join(output_dir, f"{safe_name}_cv_metrics.csv")
    metrics_df.to_csv(metrics_csv_path)

    # 80/20 train-test split for visualizations
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=3,
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
    ax.set_title(f"Confusion matrix (80/20 split) - {model_name}")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{safe_name}_cm_8020.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")   # <– use fig

    plt.show()

    # ROC curve + AUC
    auc_test = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"ROC curve - {model_name} (AUC on test = {auc_test:.3f})")

    roc_path = os.path.join(output_dir, f"{safe_name}_roc_8020.png")
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")  # <– use fig

    plt.show()

    return metrics_cv


def evaluation_on_holdout(model, X_holdout, y_holdout,
                          model_name="Model", output_dir="outputs"):

    os.makedirs(output_dir, exist_ok=True)
    safe_name = _safe_model_name(model_name)

    y_pred = model.predict(X_holdout)
    y_proba = model.predict_proba(X_holdout)[:, 1]

    auc = roc_auc_score(y_holdout, y_proba)
    acc = accuracy_score(y_holdout, y_pred)
    prec = precision_score(y_holdout, y_pred)
    rec = recall_score(y_holdout, y_pred)

    print("\n=== FINAL HOLDOUT SET PERFORMANCE ===")
    print(f"AUC:        {auc:.3f}")
    print(f"Accuracy:   {acc:.3f}")
    print(f"Precision:  {prec:.3f}")
    print(f"Recall:     {rec:.3f}")

    metrics_holdout = {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
    }

    json_path = os.path.join(output_dir, f"{safe_name}_holdout_metrics.json")
    csv_path = os.path.join(output_dir, f"{safe_name}_holdout_metrics.csv")

    with open(json_path, "w") as f:
        json.dump(metrics_holdout, f, indent=4)

    pd.DataFrame([metrics_holdout]).to_csv(csv_path, index=False)

    labels = [0, 1]
    cm = confusion_matrix(y_holdout, y_pred, labels=labels)

    n = cm.shape[0]
    correct_mask = np.eye(n)
    cmap = ListedColormap(['#912dcf', '#e0d609'])

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
    ax.set_title(f"Confusion Matrix (Holdout) - {model_name}")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{safe_name}_cm_holdout.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")

    plt.show()

    return metrics_holdout