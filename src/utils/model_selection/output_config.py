import os
from datetime import datetime
from src.constants.translators import DEFAULT_CONFIG

def _return_csv(final_dataset_name, scores_dataframe, extra_metrics=None, filter_csv=None, save_csv=False):
    """
    Filter and save the results DataFrame as a CSV file.

    Parameters:
    -----------
    final_dataset_name : str
        The base name for the output CSV file.
    scores_dataframe : pandas.DataFrame
        The DataFrame containing the results to be processed.
    extra_metrics : list, optional
        List of additional metrics to include in the CSV. Defaults to None.
    filter_csv : dict, optional
        Dictionary specifying filtering criteria for the DataFrame.
        Format: {"metric": {"h": high_threshold, "l": low_threshold}}.
        Defaults to None.
    save_csv : bool, optional
        Whether to save the final DataFrame as a CSV file. Defaults to False.

    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with specified columns removed and filters applied.

    Notes:
    ------
    - Drops irrelevant columns like "Classif_rates", "Clf", "Hyp", "Sel_feat".
    - Optionally applies high/low filters to the metrics specified in `filter_csv`.
    - Saves the processed DataFrame to a file if `save_csv` is True.
    """
    # Define the output file path
    results_path = f"{final_dataset_name}_outerloops_results.csv"

    # Columns to drop by default
    cols_drop = ["Classif_rates","Clf", "Hyp", "Sel_feat"]

    # Additional columns to drop if present
    if "Out_scor" in scores_dataframe.columns:
        cols_drop.append("Out_scor")
    if "Scoring" in scores_dataframe.columns:
        cols_drop.append("Scoring")

    # Include extra metrics in columns to drop
    if extra_metrics is not None:
        cols_drop.extend(extra_metrics)

    # Drop specified columns from the DataFrame
    statistics_dataframe = scores_dataframe.drop(cols_drop, axis=1)

    # Apply filtering if specified
    if filter_csv is not None:
        try:
            for metric, bounds in filter_csv.items():
                if "h" in bounds:
                    statistics_dataframe = statistics_dataframe[statistics_dataframe[metric] >= bounds["h"]]
                if "l" in bounds:
                    statistics_dataframe = statistics_dataframe[statistics_dataframe[metric] <= bounds["l"]]
        except Exception as e:
            print(f"An error occurred while filtering the final CSV file: {e}\nThe final CSV file will not be filtered.")

    # Save the processed DataFrame to a CSV file if required
    if save_csv:
        statistics_dataframe.to_csv(results_path, index=False)
        print(f"Statistics results saved to {results_path}")

    return statistics_dataframe

def _file_name(config):
    """
    Generate a unique file name based on the configuration and current timestamp.

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing parameters and their values.

    Returns:
    --------
    str
        Generated file name string.

    Notes:
    ------
    - Compares each config key to `DEFAULT_CONFIG` to detect custom values.
    - Appends the current date and time for uniqueness.
    """
    name_add = ""

    for conf, value in config.items():
        if conf in DEFAULT_CONFIG and config[conf] != DEFAULT_CONFIG[conf]:
            name_add += f"_{conf}_{value}"

    # Append timestamp
    name_add += f"_{datetime.now().strftime('%Y%m%d_%H%M')}"

    return name_add

def _name_outputs(config, results_dir, csv_dir):
    """
    Construct the full path for the output file name.

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing parameters and their values.
    results_dir : str
        Directory where the results will be saved.
    csv_dir : str
        Path to the input CSV file for extracting dataset name.

    Returns:
    --------
    str
        Full path to the output results file.
    """
    try:
        # Extract dataset name and append custom configuration
        dataset_name = _set_result_csv_name(csv_dir)
        name_add = _file_name(config)
        results_name = f"{dataset_name}_{name_add}_{config['model_selection_type']}"
        final_dataset_name = os.path.join(results_dir, results_name)
    except Exception as e:
        # Fallback to a generic results name if errors occur
        name_add = _file_name(config)
        results_name = f"results_{name_add}_{config['model_selection_type']}"
        final_dataset_name = os.path.join(results_dir, results_name)

    return final_dataset_name

def _set_result_csv_name(csv_dir):
    """
    Extract the base name of the dataset from the CSV file path.

    Parameters:
    -----------
    csv_dir : str
        Path to the input CSV file.

    Returns:
    --------
    str
        Base name of the dataset, without file extension.

    Notes:
    ------
    - Strips the file extension from the CSV file name.
    """
    data_name = os.path.basename(csv_dir).split(".")[0]
    return data_name
