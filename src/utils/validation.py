from sklearn.metrics import get_scorer_names

def _validate_scoring(scoring: str) -> None:
    """
    Validate the scoring metric provided as input.

    :param scoring: The scoring metric as a string.
    :type scoring: str
    :raises ValueError: If the scoring metric is invalid.
    """
    valid_metrics = list(get_scorer_names()) + ["specificity"]
    if scoring not in valid_metrics:
        raise ValueError(
            f"Invalid scoring metric: {scoring}. Select one of the following: {valid_metrics}"
        )

def _normalize_config_value(config, key, valid_values, default=None):
    """
    Normalize a configuration value to ensure it's valid.

    :param config: The configuration dictionary.
    :type config: dict
    :param key: The key to normalize in the config.
    :type key: str
    :param valid_values: List of valid values for the key.
    :type valid_values: list
    :param default: Default value if the key's value is invalid or not provided.
    :type default: any
    :return: None (modifies config in place).
    :rtype: None
    """
    if config.get(key) not in valid_values:
        print(f"Invalid or missing value for {key}. Setting default to {default}")
        config[key] = default

def _process_missing_values(config, X):
    """
    Handle missing values in the dataset based on the provided configuration.

    :param config: The configuration dictionary.
    :type config: dict
    :param X: The dataset.
    :type X: pandas.DataFrame
    :return: None (modifies config in place).
    :rtype: None
    """
    if config.get("missing_values_method") == "drop":
        print("Missing values cannot be dropped. Using 'median' as default.")
        config["missing_values_method"] = "median"

    if X.isnull().values.any():
        print(
            f"Dataset contains NaN values. Using {config['missing_values_method']} for missing values."
        )

def _process_features(config, X, main_type):
    """
    Process and validate the number of features in the configuration.

    :param config: The configuration dictionary.
    :type config: dict
    :param X: The dataset.
    :type X: pandas.DataFrame
    :param main_type: The type of validation process ('rncv' or other).
    :type main_type: str
    :return: None (modifies config in place).
    :rtype: None
    """
    num_features = config.get("num_features")
    
    if main_type in ['rncv', 'rcv']:
        if isinstance(num_features, int):
            config["num_features"] = [num_features]
        elif num_features is None:
            config["num_features"] = [X.shape[1]]
        elif not isinstance(num_features, list):
            raise ValueError("num_features must be an integer, a list, or None.")
    else:
        if num_features is None or num_features > X.shape[1]:
            config["num_features"] = X.shape[1]
        elif not isinstance(num_features, int):
            raise ValueError("num_features must be an integer or None.")

def _validate_extra_metrics(config):
    """
    Validate and normalize extra metrics in the configuration.

    :param config: The configuration dictionary.
    :type config: dict
    :return: None (modifies config in place).
    :rtype: None
    """
    extra_metrics = config.get("extra_metrics", [])

    if not isinstance(extra_metrics, list):
        extra_metrics = [extra_metrics]

    for metric in extra_metrics:
        _validate_scoring(metric)

    config["extra_metrics"] = extra_metrics

def _configure_classifiers(config, available_clfs):
    """
    Configure the classifiers based on the provided include/exclude lists.

    :param config: The configuration dictionary.
    :type config: dict
    :param available_clfs: Available classifiers.
    :type available_clfs: dict
    :return: None (modifies config in place).
    :rtype: None
    """
    include_classes = config.get("search_on")
    exclude_classes = config.get("exclude", [])

    if include_classes:
        config["clfs"] = [clf for clf in include_classes if clf in available_clfs]
    else:
        config["clfs"] = [clf for clf in available_clfs if clf not in exclude_classes]

def _validation(config, main_type, X, csv_dir, label, available_clfs):
    """
    Validate and preprocess the configuration dictionary for a machine learning pipeline.

    :param config: Configuration dictionary.
    :type config: dict
    :param main_type: The type of process ('rncv', 'rcv', etc.).
    :type main_type: str
    :param X: Input dataset.
    :type X: pandas.DataFrame
    :param csv_dir: Path to the CSV file.
    :type csv_dir: str
    :param label: Target label name.
    :type label: str
    :param available_clfs: Available classifiers.
    :type available_clfs: dict
    :return: Validated and normalized configuration dictionary.
    :rtype: dict
    """
    # Handle missing values
    _process_missing_values(config, X)

    # Validate and normalize features
    _process_features(config, X, main_type)

    # Validate scoring and metrics
    if main_type == "rncv":
        _validate_scoring(config.get("outer_scoring"))
        _validate_scoring(config.get("inner_scoring"))
        if config['outer_scoring'] not in config['extra_metrics']:
            config['extra_metrics'].insert(0, config['outer_scoring'])
    else:
        _validate_scoring(config.get("scoring"))
        if config['scoring'] not in config['extra_metrics']:
            config['extra_metrics'].insert(0, config['scoring'])

    _validate_extra_metrics(config)

    # Normalize class balancing
    _normalize_config_value(
        config,
        "class_balance",
        valid_values=["smote", "borderline_smote", "tomek", None],
        default=None,
    )

    # Configure classifiers
    _configure_classifiers(config, available_clfs)

    # Add additional configurations
    config["dataset_name"] = csv_dir
    config["dataset_label"] = label
    config["features_name"] = (
        None if config["num_features"] == [X.shape[1]] else config["num_features"]
    )
    config["all_features"] = X.shape[1]
    config['model_selection_type'] = main_type
    return config



# from sklearn.metrics import get_scorer, confusion_matrix, get_scorer_names
# from sklearn.metrics import average_precision_score, roc_auc_score

# def _scoring_check(scoring: str) -> None:
#     """This function is used to check if the scoring string metric is valid"""
#     if (scoring not in get_scorer_names()) and (scoring != "specificity"):
#         raise ValueError(
#             f"Invalid scoring metric: {scoring}. Select one of the following: {list(get_scorer_names())} and specificity"
#         )
        
# def _validation(config, main_type, X, csv_dir, label, available_clfs):
#     """
#     This function checks the parameters of the pipeline and returns the final parameters config for the class pipeline.
#     """
#     # Missing values manipulation
#     if config['missing_values_method'] == "drop":
#         print(
#             "Values cannot be dropped at ncv because of inconsistent shapes. \nThe missing values with automaticly replaced by the median of each feature."
#         )
#         config['missing_values_method'] = "median"
#     elif (config['missing_values_method'] != "mean") and (config['missing_values_method'] != "median"):
#         raise ValueError(
#             "The missing values method should be 'mean' or 'median'."
#         )
#     if X.isnull().values.any():
#         print(
#             f"Your Dataset contains NaN values. Some estimators does not work with NaN values.\nThe {config['missing_values_method']} method will be used for the missing values manipulation.\n"
#         )
        
#     if (main_type == 'rncv') or (main_type == 'rcv'):
#         # Checks for reliability of parameters
#         if isinstance(config["num_features"], int):
#             config["num_features"] = [config["num_features"]]
#         elif isinstance(config["num_features"], list):
#             pass
#         elif config["num_features"] is None:
#             config["num_features"] = [X.shape[1]]
#         else:
#             raise ValueError("num_features must be an integer or a list or None")
#     else: 
#         if (config["num_features"] is None) or (config["num_features"] > X.shape[1]):
#             config["num_features"] = X.shape[1]
#         elif isinstance(config["num_features"], int):
#             pass
#         else:
#             raise ValueError("num_features must be an integer or None")

#     if config['extra_metrics'] is not None:
#         if type(config['extra_metrics']) is not list:
#             config['extra_metrics'] = [config['extra_metrics']]
#         for metric in config['extra_metrics']:
#             _scoring_check(metric)
        
#     if main_type == 'rncv':    
#         if config['outer_scoring'] not in config['extra_metrics']:
#             config['extra_metrics'].insert(0, config['outer_scoring'])
#         elif config['outer_scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['outer_scoring']) != 0:
#             # Remove it from its current position
#             config['extra_metrics'].remove(config['outer_scoring'])
#             # Insert it at the first index
#             config['extra_metrics'].insert(0, config['outer_scoring'])
#         elif config['outer_scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['outer_scoring']) == 0:
#             pass
#         else:
#             config['extra_metrics'] = [config['outer_scoring']]
                    
#         # Checks for reliability of parameters
#         if (config['inner_scoring'] not in get_scorer_names()) and (config['inner_scoring'] != "specificity"):
#             raise ValueError(
#                 f"Invalid inner scoring metric: {config['inner_scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
#             )
#         if (config['outer_scoring'] not in get_scorer_names()) and (config['outer_scoring'] != "specificity"):
#             raise ValueError(
#                 f"Invalid outer scoring metric: {config['outer_scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
#             )
#         for inner_selection in config['inner_selection']:
#             if inner_selection not in ["validation_score", "one_sem", "gso_1", "gso_2","one_sem_grd"]:
#                 raise ValueError(
#                     f'Invalid inner method: {inner_selection}. Select one of the following: ["validation_score", "one_sem", "one_sem_grd", "gso_1", "gso_2"]'
#                 )
                
#         if (config['parallel'] not in ['thread_per_round', 'freely_parallel']) and (config['parallel'] is not None):
#             raise ValueError(
#                 f'Invalid parallel method: {config["parallel"]}. Select one of the following: ["thread_per_round", "freely_parallel"]'
#         )
#         elif (config['parallel'] == None):
#             config['parallel'] = 'thread_per_round'
#             print('Parallel method is set to "thread_per_round"')
        
#     else:
#         if 'evaluation' not in config.keys():
#             config['evaluation'] = None
#         if 'param_grid' not in config.keys():
#             config['param_grid'] = None
#         if 'features_names_list' not in config.keys():  
#             config['features_names_list'] = None
#         if config['scoring'] not in config['extra_metrics']:
#             config['extra_metrics'].insert(0, config['scoring'])
#         elif config['scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['scoring']) != 0:
#             # Remove it from its current position
#             config['extra_metrics'].remove(config['scoring'])
#             # Insert it at the first index
#             config['extra_metrics'].insert(0, config['scoring'])
#         elif config['scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['scoring']) == 0:
#             pass
#         else:
#             config['extra_metrics'] = [config['scoring']]
        
#         if (config['scoring'] not in get_scorer_names()) and (config['scoring'] != "specificity"):
#             raise ValueError(
#                 f"Invalid scoring metric: {config['scoring']}. Select one of the following: {list(get_scorer_names())} and specificity"
#             )
            
#     config['model_selection_type'] = main_type
    
#     if config['class_balance'] not in ['smote','borderline_smote','tomek', None]:
#         raise ValueError("class_balance must be one of the following: 'smote','smotenn','adasyn','borderline_smote','tomek', or None")
#     elif config['class_balance'] == None:
#         config['class_balance'] = 'None'
#         print('No class balancing will be applied')
        
#     config['dataset_name'] = csv_dir
#     config['dataset_label'] = label
#     config['features_name'] = None if (config['num_features'] == [X.shape[1]]) or (config['num_features'] == X.shape[1]) else config['num_features']
#     config['all_features'] = X.shape[1]
            
#     if (main_type == 'rncv') or (main_type == 'rcv'):
#         # Set available classifiers
#         if config['search_on'] is not None:
#             classes = config['search_on']  # 'search_on' is a list of classifier names as strings
#             exclude_classes = [
#                 clf for clf in available_clfs.keys() if clf not in classes
#             ]
#         elif config['exclude'] is not None:
#                 exclude_classes = (
#                 config['exclude']  # 'exclude' is a list of classifier names as strings
#             )
#         else:
#             exclude_classes = []

#         # Filter classifiers based on the exclude_classes list
#         clfs = [clf for clf in available_clfs.keys() if clf not in exclude_classes]
#         config["clfs"] = clfs
#     else: 
#         if config['param_grid'] is None:
#             config['param_grid'] = 'None'
#         else:
#             config['param_grid'] = str(config['param_grid'])
#         if config['features_names_list'] is None:
#             config['features_names_list'] = 'None'
#         else:
#             config['features_names_list'] = str(config['features_names_list'])
#     return config 