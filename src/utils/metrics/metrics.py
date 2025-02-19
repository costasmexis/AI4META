from typing import Dict, List, Union, Any
import numpy as np
from scipy.stats import sem
from sklearn.metrics import (
    get_scorer,
    confusion_matrix,
    get_scorer_names,
    average_precision_score,
    roc_auc_score
)

def _specificity_scorer(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the specificity score (true negative rate) for a classifier.

    Specificity measures the proportion of actual negatives correctly identified.
    It is calculated as: TN / (TN + FP), where:
    - TN: True Negatives
    - FP: False Positives

    Parameters
    ----------
    estimator : Any
        Trained classifier with predict method
    X : np.ndarray
        Input features
    y : np.ndarray
        True labels

    Returns
    -------
    float
        Specificity score in range [0, 1]
    """
    y_pred = estimator.predict(X)
    tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
    return tn / (tn + fp)

def _calculate_metrics(
    metrics: List[str],
    results: Dict[str, List[float]],
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, List[float]]:
    """
    Calculate multiple performance metrics for a classifier.

    This function computes specified metrics for a trained classifier using test data.
    It handles both standard sklearn metrics and custom metrics like specificity.
    For probability-based metrics (roc_auc, average_precision), it uses predict_proba
    when available.

    Parameters
    ----------
    metrics : List[str]
        List of metric names to calculate
    results : Dict[str, List[float]]
        Dictionary to store results, with metric names as keys
    clf : Any
        Trained classifier with predict method
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        True test labels

    Returns
    -------
    Dict[str, List[float]]
        Updated results dictionary with new metric scores

    Raises
    ------
    AttributeError
        If probability-based metrics are requested but classifier doesn't support predict_proba
    """
    for metric in metrics:
        if metric == 'specificity':
            results[metric].append(_specificity_scorer(clf, X_test, y_test))
            continue

        try:
            # Try standard sklearn scoring
            score = get_scorer(metric)(clf, X_test, y_test)
            results[metric].append(score)
        except AttributeError:
            # Handle probability-based metrics
            if metric in ['roc_auc', 'average_precision']:
                if not hasattr(clf, 'predict_proba'):
                    raise AttributeError(
                        f"Model {type(clf).__name__} doesn't support predict_proba, "
                        f"which is required for {metric}"
                    )
                
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                
                if metric == 'roc_auc':
                    score = roc_auc_score(y_test, y_pred_proba)
                else:  # average_precision
                    score = average_precision_score(y_test, y_pred_proba)
                    
                results[metric].append(score)
            else:
                raise  # Re-raise unexpected AttributeErrors

    return results