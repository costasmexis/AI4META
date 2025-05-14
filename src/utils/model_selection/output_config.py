import os
from typing import Optional, Dict, List, Any
import pandas as pd
# from datetime import datetime
# from src.constants.translators import DEFAULT_CONFIG

def _return_csv(
    final_dataset_name: str,
    scores_dataframe: pd.DataFrame,
    extra_metrics: Optional[List[str]] = None,
    # filter_csv: Optional[Dict[str, Dict[str, float]]] = None,
    save_csv: bool = False
) -> pd.DataFrame:
    """
    Process and optionally save a results DataFrame with filtering options.

    This function handles the post-processing of model evaluation results, including
    column filtering, metric thresholding, and optional CSV export.

    Parameters
    ----------
    final_dataset_name : str
        Base name for the output CSV file
    scores_dataframe : pd.DataFrame
        DataFrame containing the model evaluation results
    extra_metrics : list[str], optional
        Additional metrics to include in processing
    filter_csv : dict, optional
        Filtering criteria for metrics. Format:
        {"metric_name": {"h": high_threshold, "l": low_threshold}}
    save_csv : bool, default=False
        Whether to save the processed DataFrame to CSV
    """
    # Construct output file path
    results_path = f"{final_dataset_name}_outerloops_results.csv"

    # Define columns to remove
    cols_to_drop = ["Classif_rates", "Clf", "Hyp", "Sel_feat"]

    # Add scoring-related columns if present
    if "Scoring" in scores_dataframe.columns:
        cols_to_drop.append("Scoring")

    # Add extra metrics to columns for removal
    if extra_metrics:
        cols_to_drop.extend(extra_metrics)

    # Process DataFrame
    statistics_dataframe = scores_dataframe.drop(cols_to_drop, axis=1)

    # # Apply metric filtering if specified
    # if filter_csv:
    #     try:
    #         for metric, bounds in filter_csv.items():
    #             if "h" in bounds:  # Apply high threshold
    #                 statistics_dataframe = statistics_dataframe[
    #                     statistics_dataframe[metric] >= bounds["h"]
    #                 ]
    #             if "l" in bounds:  # Apply low threshold
    #                 statistics_dataframe = statistics_dataframe[
    #                     statistics_dataframe[metric] <= bounds["l"]
    #                 ]
    #     except Exception as e:
    #         print(f"Error during CSV filtering: {e}\nProceeding without filters.")

    # Save to CSV if requested
    if save_csv:
        statistics_dataframe.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

    return statistics_dataframe

# def _file_name(config: Dict[str, Any]) -> str:
#     """
#     Generate a unique file name based on configuration and timestamp.

#     Creates a string combining non-default configuration values and current
#     timestamp for unique identification of result files.

#     Parameters
#     ----------
#     config : dict
#         Configuration dictionary to compare against defaults
#     """
#     # Build name components list
#     name_components = []

#     # Add non-default configurations
#     for param, value in config.items():
#         if param in DEFAULT_CONFIG and config[param] != DEFAULT_CONFIG[param]:
#             name_components.append(f"{param}_{value}")

#     # Add timestamp
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M')
#     name_components.append(timestamp)

#     # Join components
#     return "_".join(name_components)

# def _name_outputs(
#     config: Dict[str, Any],
#     results_dir: str,
#     csv_dir: str
# ) -> str:
#     """
#     Construct complete output file path for results.

#     Parameters
#     ----------
#     config : dict
#         Configuration dictionary
#     results_dir : str
#         Directory path for results storage
#     csv_dir : str
#         Path to input CSV file
#     """
#     try:
#         # Generate dataset-specific name
#         dataset_name = _set_result_csv_name(csv_dir)
#         name_suffix = _file_name(config)
#         results_name = f"{dataset_name}_{name_suffix}_{config['model_selection_type']}"
#         return os.path.join(results_dir, results_name)
#     except Exception as e:
#         # Fall back to generic naming
#         print(f"Warning: Using generic name due to error: {e}")
#         name_suffix = _file_name(config)
#         results_name = f"results_{name_suffix}_{config['model_selection_type']}"
#         return os.path.join(results_dir, results_name)

# def _set_result_csv_name(csv_dir: str) -> str:
#     """
#     Extract base name from CSV file path.

#     Parameters
#     ----------
#     csv_dir : str
#         Path to CSV file
#     """
#     return os.path.basename(csv_dir).split(".")[0]
