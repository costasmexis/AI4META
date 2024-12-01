import os 
from datetime import datetime

def _return_csv( final_dataset_name, scores_dataframe, extra_metrics=None, filter_csv=None):
    """
    Function to save the results to a csv file

    Parameters
    ----------
    final_dataset_name : str
        Name of the dataset
    scores_dataframe : DataFrame
        DataFrame containing the results
    extra_metrics : list, optional
        List of extra metrics to include in the csv file
    filter_csv : dict, optional
        Dictionary containing the filters to apply to the csv file

    Returns
    -------
    DataFrame
        The filtered DataFrame
    """
    results_path = f"{final_dataset_name}_outerloops_results.csv"
    cols_drop = ["Classif_rates", "Clf", "Hyp", "Sel_feat"]
    if extra_metrics is not None:
        for metric in extra_metrics:
            cols_drop.append(f"{metric}") 
    statistics_dataframe = scores_dataframe.drop(cols_drop, axis=1)
    if filter_csv is not None:
        try:
            # Apply filters to the csv file
            for mtrc_stat in filter_csv:
                if 'h' in filter_csv[mtrc_stat]:
                    statistics_dataframe = statistics_dataframe[statistics_dataframe[mtrc_stat] >= filter_csv[mtrc_stat]['h']]
                elif 'l' in filter_csv[mtrc_stat]:
                    statistics_dataframe = statistics_dataframe[statistics_dataframe[mtrc_stat] <= filter_csv[mtrc_stat]['l']]
        except Exception as e:
            print(f'An error occurred while filtering the final csv file: {e}\nThe final csv file will not be filtered.')
    statistics_dataframe.to_csv(results_path, index=False)
    print(f"Statistics results saved to {results_path}")
    return statistics_dataframe

def _file_name(config):
    """
    Function to set the name of the result nested cv and rcv_accel file with respect to the dataset name

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of the experiment

    Returns
    -------
    str
        The name of the result file
    """
    # Default values of the configuration
    default_values = {
        "rounds": 10,
        "n_trials": 100,
        "feature_selection_type": "mrmr",
        "feature_selection_method": "chi2",
        "inner_scoring": "matthews_corrcoef",
        "outer_scoring": "matthews_corrcoef",
        "inner_splits": 5,
        "outer_splits": 5,
        "normalization": "minmax",
        "class_balance": "auto",    
        "sfm": False,
        "missing_values": "median",
        "num_features": None,
        "scoring": "matthews_corrcoef",
        "splits": 5
        
    }
    # Generate the name of the result file
    name_add = ""
    for conf in config:
        if conf in default_values.keys():
            if config[conf] != default_values[conf]:
                name_add += f"_{conf}_{config[conf]}"
    name_add += f"_{datetime.now().strftime('%Y%m%d_%H%M')}"
    return name_add

def _name_outputs( config, results_dir, csv_dir):
    """ Function to set the name of the result nested cv file with respect to the dataset name """
    try:
        dataset_name = _set_result_csv_name(csv_dir)
        name_add  = _file_name(config)
        results_name = f"{dataset_name}_{name_add}_{config['model_selection_type']}"
        final_dataset_name = os.path.join(results_dir, results_name)
    except Exception as e:
        name_add = _file_name(config)
        results_name = f"results_{name_add}_{config['model_selection_type']}"
        final_dataset_name = os.path.join(
            results_dir, results_name
        )
    return final_dataset_name

def _set_result_csv_name( csv_dir):
    """This function is used to set the name of the result nested cv file with respect to the dataset name"""
    # Split the name of the dataset with the ".csv" part and keep the first part
    data_name = os.path.basename(csv_dir).split(".")[0]
    print(data_name)
    return data_name
