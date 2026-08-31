import numpy as np
from loguru import logger
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,)

def evaluate_models(
        X_test, y_test, models
):
    """
    Evaluate the trained models on the test set.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
        models (dict): Dictionary of trained models.

    Returns:
        dict: A dictionary containing evaluation metrics for each model.
    """
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:,1]

        best_treshold = None
        best_f1 = -1
        best_precision = None
        best_recall = None

        for threshold in np.range(0.05, 0.50, 0.01):
            y_pred = (y_proba >= 0.05).astype(int)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_treshold = threshold
                best_precision = precision
                best_recall = recall
        roc_auc = roc_auc_score(y_test, y_proba,)

        logger.info(f"Model: {name}")
        logger.info(f"Best Threshold: {best_treshold:.2f}")
        logger.info(f"Precision: {best_precision:.4f}")
        logger.info(f"Recall: {best_recall:.4f}")
        logger.info(f"F1 Score: {best_f1:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")