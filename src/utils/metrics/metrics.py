import numpy as np
from scipy.stats import sem
from sklearn.metrics import get_scorer, confusion_matrix, get_scorer_names
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.translators import METRIC_ADDREVIATIONS
from src.utils.statistics.bootstrap_ci import _calc_ci_btstrp

def _calculate_metrics(metrics, results, clf, X_test, y_test):
    for metric in metrics:
        if metric == 'specificity':
            results[f"{metric}"].append(
                _specificity_scorer(clf, X_test, y_test)
            )
        else:
            try:   
                results[f"{metric}"].append(
                    get_scorer(metric)(clf, X_test, y_test)
                )
            except AttributeError:
                # Handle metrics like roc_auc and average_precision explicitly
                if metric in ['roc_auc', 'average_precision']:
                    if hasattr(clf, 'predict_proba'):
                        # Use decision_function if available
                        y_pred = clf.predict_proba(X_test)[:, 1]
                    else:
                        raise AttributeError(
                            f"Model {type(clf).__name__} does not support `predict_proba`, "
                            f"which are required for {metric}."
                        )

                    # Compute the score using the selected y_pred
                    if metric == 'roc_auc':
                        score = roc_auc_score(y_test, y_pred)
                    elif metric == 'average_precision':
                        score = average_precision_score(y_test, y_pred)

                results[f"{metric}"].append(score)
    return results
    
def _specificity_scorer(estimator, X, y):
    """_This function is used to calculate the specificity score"""
    y_pred = estimator.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

