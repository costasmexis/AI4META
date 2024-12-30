import numpy as np
from scipy.stats import sem
from sklearn.metrics import get_scorer, confusion_matrix, get_scorer_names
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.translators import METRIC_ADDREVIATIONS
from src.utils.statistics.bootstrap_ci import _calc_ci_btstrp

def _calculate_metrics(metrics, results, clf, X_test, y_test):
    """
    Calculate specified metrics for a given model and append them to the results dictionary.

    Parameters:
    -----------
    metrics : list
        List of metric names to calculate.
    results : dict
        Dictionary to store the calculated metric values.
    clf : object
        Trained classifier to evaluate.
    X_test : array-like
        Feature data for testing.
    y_test : array-like
        True labels for testing.

    Returns:
    --------
    dict
        Updated results dictionary containing calculated metric values.

    Notes:
    ------
    - Handles specific metrics like 'specificity', 'roc_auc', and 'average_precision' separately.
    - Uses `get_scorer` for standard metrics.
    """
    for metric in metrics:
        if metric == 'specificity':
            # Calculate specificity
            results[f"{metric}"].append(_specificity_scorer(clf, X_test, y_test))
        else:
            try:
                # Use sklearn's get_scorer for standard metrics
                results[f"{metric}"].append(get_scorer(metric)(clf, X_test, y_test))
            except AttributeError:
                # Handle non-standard metrics explicitly
                if metric in ['roc_auc', 'average_precision']:
                    if hasattr(clf, 'predict_proba'):
                        y_pred = clf.predict_proba(X_test)[:, 1]
                    else:
                        raise AttributeError(
                            f"Model {type(clf).__name__} does not support `predict_proba`, required for {metric}."
                        )

                    # Calculate the metric explicitly
                    if metric == 'roc_auc':
                        score = roc_auc_score(y_test, y_pred)
                    elif metric == 'average_precision':
                        score = average_precision_score(y_test, y_pred)

                    results[f"{metric}"].append(score)
    return results

def _specificity_scorer(estimator, X, y):
    """
    Calculate the specificity score for a given estimator.

    Specificity is calculated as:
    TN / (TN + FP)

    Parameters:
    -----------
    estimator : object
        Trained classifier.
    X : array-like
        Feature data for prediction.
    y : array-like
        True labels.

    Returns:
    --------
    float
        Specificity score.
    """
    y_pred = estimator.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity
