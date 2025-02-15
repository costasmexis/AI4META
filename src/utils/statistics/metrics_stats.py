import numpy as np
from scipy.stats import sem
from src.constants.translators import METRIC_ADDREVIATIONS
from src.utils.statistics.bootstrap_ci import _calc_ci_btstrp
import pandas as pd

def _calc_metrics_stats(extra_metrics, results, indices):
    """
    Calculate statistical summaries for specified metrics and store them in the results dictionary.

    This function processes a list of metrics, computes summary statistics such as mean, standard deviation,
    median, and confidence intervals, and stores the results in a structured format for further analysis.

    Parameters:
    -----------
    extra_metrics : list
        List of metrics to calculate statistics for.
    results : list of dict
        List where each dictionary corresponds to the statistics for a model or configuration.
        The last entry in this list will be updated with the computed metrics.
    indices : pandas.DataFrame
        DataFrame containing the metric values for each configuration or model.

    Returns:
    --------
    results : list of dict
        Updated results list with calculated statistics for the specified metrics.

    Notes:
    ------
    - Metric abbreviations are retrieved from `METRIC_ADDREVIATIONS` for compact naming.
    - Confidence intervals for both mean and median are computed using bootstrapping.
    - Rounded values (up to 3 decimal places) are used for consistency.
    """
    # Metric abbreviation mapping
    metric_abbreviations = METRIC_ADDREVIATIONS

    # Iterate over each metric and calculate statistics
    for metric in extra_metrics:
        # Get the abbreviated metric name
        qck_mtrc = metric_abbreviations.get(metric, metric)

        # Extract metric values from indices
        if metric not in indices:
            raise KeyError(f"Metric '{metric}' not found in the indices DataFrame.")
        
        metric_values = indices[metric].values

        # Ensure results[-1] exists
        if not results or not isinstance(results[-1], dict):
            raise ValueError("Invalid results structure. The last entry must be a dictionary.")

        # Store the raw metric values in the results
        results[-1][metric] = metric_values

        # Calculate basic statistical summaries
        results[-1][f"{qck_mtrc}_mean"] = round(np.mean(metric_values), 3)
        results[-1][f"{qck_mtrc}_std"] = round(np.std(metric_values), 3)
        results[-1][f"{qck_mtrc}_sem"] = round(sem(metric_values), 3)

        # Calculate and store the median
        results[-1][f"{qck_mtrc}_med"] = round(np.median(metric_values), 3)

        # Bootstrap confidence intervals for median and mean
        lomed, upmed = _calc_ci_btstrp(metric_values, central_tendency="median")
        lomean, upmean = _calc_ci_btstrp(metric_values, central_tendency="mean")

        # Store confidence intervals in the results
        results[-1][f"{qck_mtrc}_lomean"] = round(lomean, 3)
        results[-1][f"{qck_mtrc}_upmean"] = round(upmean, 3)
        results[-1][f"{qck_mtrc}_lomed"] = round(lomed, 3)
        results[-1][f"{qck_mtrc}_upmed"] = round(upmed, 3)

    return results


def final_model_stats(metrics_df):
    """
    Calculate statistics for a final model and store them in a dataframe.

    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame containing the metric values for the final model.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated statistics for the final model.
    """
    # get column names
    columns = metrics_df.columns
    # add to the first index the "type" column
    columns = ["type"] + list(columns)
    # create a new stats dataframe
    stat_df = pd.DataFrame(columns=columns)
    # to the column "type" add values 'mean', 'std' and 'sem'
    stat_df['type'] = ['mean', 'std', 'sem']
    
    # for every column in the metrics dataframe calculate the statistics
    for col in columns[1:]:
        # Convert the list in the cell to a numpy array
        values = np.array(metrics_df[col].iloc[0])  # since we have only one row
        
        # Calculate statistics
        mean_val = round(np.mean(values), 3)
        std_val = round(np.std(values), 3)
        sem_val = round(sem(values), 3)
        
        # Assign values to the stat_df
        stat_df[col] = [mean_val, std_val, sem_val]
    
    return stat_df