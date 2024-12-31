from src.data.dataloader import DataLoader

def _preprocess(X, y, num_feature2_use, config, train_index=None, test_index=None, features_names_list=None):
    """
    Preprocess the dataset by normalizing, handling missing values, and applying feature selection.

    Parameters:
    -----------
    X : pandas.DataFrame
        The input features.
    y : pandas.Series or numpy.ndarray
        The target labels.
    num_feature2_use : int or None
        Number of features to select. If None, all features are retained.
    config : dict
        Configuration dictionary containing preprocessing settings.
    train_index : array-like or None, optional
        Indices for the training set. If None, uses the entire dataset.
    test_index : array-like or None, optional
        Indices for the testing set. If None, uses the entire dataset.
    features_names_list : list or None, optional
        List of feature names to retain. If None, all features are used.

    Returns:
    --------
    tuple
        X_train_selected : pandas.DataFrame
            The preprocessed training set with selected features.
        X_test_selected : pandas.DataFrame
            The preprocessed testing set with selected features.
        num_feature : int or str
            The number of features selected or 'none' if all features are retained.

    Raises:
    -------
    ValueError
        If the number of features to select is invalid.

    Notes:
    ------
    - Applies normalization, missing value handling, and feature selection sequentially.
    - Supports percentile-based and standard feature selection methods.
    """
    data_loader = DataLoader(label=config["dataset_label"], csv_dir=config["dataset_name"])

    # Split the data into training and testing sets
    if train_index is None and test_index is None:
        X_tr, X_te = X.copy(), X.copy()
        y_train = y.copy()
    else:
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y[train_index], y[test_index]

    # Retain specific features if provided
    if features_names_list is not None:
        X_tr = X_tr[features_names_list]
        X_te = X_te[features_names_list]

    # Normalize the training and testing sets
    X_train, X_test = data_loader.normalize(
        X=X_tr, train_test_set=True, X_test=X_te, method=config["normalization"]
    )

    # Handle missing values for training and testing sets
    X_train = data_loader.missing_values(data=X_train, method=config["missing_values_method"])
    X_test = data_loader.missing_values(data=X_test, method=config["missing_values_method"])

    # Apply feature selection based on the specified method
    if config["feature_selection_type"] != "percentile":
        if isinstance(num_feature2_use, int):
            if num_feature2_use < X_train.shape[1]:
                selected_features = data_loader.feature_selection(
                    X=X_train,
                    y=y_train,
                    method=config["feature_selection_type"],
                    num_features=num_feature2_use,
                    inner_method=config["feature_selection_method"],
                )
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]
                num_feature = num_feature2_use
            elif num_feature2_use == X_train.shape[1]:
                X_train_selected = X_train
                X_test_selected = X_test
                num_feature = "none"
            else:
                raise ValueError(
                    "num_features must be an integer less than the number of features in the dataset"
                )
    elif config["feature_selection_type"] == "percentile":
        if isinstance(num_feature2_use, int):
            if num_feature2_use in [100, X.shape[1]]:
                X_train_selected = X_train
                X_test_selected = X_test
                num_feature = "none"
            elif 0 < num_feature2_use < 100:
                selected_features = data_loader.feature_selection(
                    X=X_train,
                    y=y_train,
                    method=config["feature_selection_type"],
                    inner_method=config["feature_selection_method"],
                    num_features=num_feature2_use,
                )
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]
                num_feature = num_feature2_use
            else:
                raise ValueError(
                    "num_features must be an integer between 1 and 100 (inclusive)"
                )
    else:
        raise ValueError("Invalid feature selection type or num_features input.")

    return X_train_selected, X_test_selected, num_feature
