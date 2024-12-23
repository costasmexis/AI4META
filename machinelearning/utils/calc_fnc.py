import numpy as np
import shap
from scipy.stats import sem
from sklearn.metrics import get_scorer, confusion_matrix, get_scorer_names
from sklearn.metrics import average_precision_score, roc_auc_score
from .translators import METRIC_ADDREVIATIONS

def _calculate_metrics(metrics, results, clf, X_test, y_test):
    for metric in metrics:
        if metric == 'specificity':
            results[f"{metric}"].append(
                _specificity_scorer(clf, X_test, y_test)
            )
        else:
            # print(res_model)
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

def _calc_ci_btstrp(data, type='median'):
    """
    Calculate the confidence interval of the mean or median using bootstrapping.

    Args:
        data (array-like): Input data to calculate the confidence interval for.
        type (str): Type of central tendency to compute ('mean' or 'median'). Defaults to 'median'.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    ms = []
    for _ in range(1000):
        # Generate a bootstrap sample
        sample = np.random.choice(data, size=len(data), replace=True)
        
        # Compute the desired central tendency
        if type == 'median':
            ms.append(np.median(sample))
        elif type == 'mean':
            ms.append(np.mean(sample))
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = np.percentile(ms, (1 - 0.95) / 2 * 100)
    upper_bound = np.percentile(ms, (1 + 0.95) / 2 * 100)
    
    return lower_bound, upper_bound

def _calc_shap(self, X_train, X_test, model):
    try:
        explainer = shap.Explainer(model, X_train)
    except TypeError as e:
        if (
            "The passed model is not callable and cannot be analyzed directly with the given masker!"
            in str(e)
        ):
            print(
                "Switching to predict_proba due to compatibility issue with the model."
            )
            explainer = shap.Explainer(lambda x: model.predict_proba(x), X_train)
        else:
            raise TypeError(e)
    try:
        if self.best_estimator.__class__.__name__ in ["LGBMClassifier", "CatBoostClassifier","RandomForestClassifier"]:
            shap_values = explainer(X_test, check_additivity=False)
        else: 
            shap_values = explainer(X_test)
    except ValueError:
        num_features = X_test.shape[1]
        max_evals = 2 * num_features + 1
        shap_values = explainer(X_test, max_evals=max_evals)

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    else:
        pass
    return shap_values
 
def _calc_metrics_stats( extra_metrics, results, indices):
    """
    Add renamed metrics to the results dataframe.

    Parameters
    ----------
    extra_metrics : list
        List of extra metrics to be added.
    results : list
        The results dataframe to which the metrics will be added.
    indices : DataFrame
        DataFrame containing the metric values.

    Returns
    -------
    list
        Updated results with renamed metrics.
    """
    # Metric abbreviation mapping
    metric_abbreviations = METRIC_ADDREVIATIONS
    
    # Iterate over each metric and calculate statistics
    for metric in extra_metrics:
        # Get the abbreviated metric name
        qck_mtrc = metric_abbreviations[f"{metric}"]
        # Extract metric values from indices
        metric_values = indices[f"{metric}"].values

        # Store the raw metric values in the results
        results[-1][f"{metric}"] = metric_values

        # Calculate mean, standard deviation, and standard error of the mean
        results[-1][f"{qck_mtrc}_mean"] = round(np.mean(metric_values), 3)
        results[-1][f"{qck_mtrc}_std"] = round(np.std(metric_values), 3)
        results[-1][f"{qck_mtrc}_sem"] = round(sem(metric_values), 3)

        # Calculate and store the median
        results[-1][f"{qck_mtrc}_med"] = round(np.median(metric_values), 3)
        # Bootstrap confidence intervals for median and mean
        lomed, upmed = _calc_ci_btstrp(metric_values, type='median')
        lomean, upmean = _calc_ci_btstrp(metric_values, type='mean')

        results[-1][f"{qck_mtrc}_lomean"] = round(lomean, 3)
        results[-1][f"{qck_mtrc}_upmean"] = round(upmean, 3)    
        results[-1][f"{qck_mtrc}_lomed"] = round(lomed, 3)
        results[-1][f"{qck_mtrc}_upmed"] = round(upmed, 3)
    
    return results