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