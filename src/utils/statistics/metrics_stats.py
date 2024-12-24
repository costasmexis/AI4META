import numpy as np
from scipy.stats import sem
from src.utils.translators import METRIC_ADDREVIATIONS
from src.utils.statistics.bootstrap_ci import _calc_ci_btstrp

def _calc_metrics_stats( extra_metrics, results, indices):
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