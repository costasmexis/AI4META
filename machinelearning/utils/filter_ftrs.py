from dataloader import DataLoader

def _filter_features(train_index, test_index, X, y, num_feature2_use, config):
    """
    This function filters the features using the selected model.
    Returns the filtered train and test sets indexes - including the normalized data and the missing values - and the selected features.
    
    Parameters
    ----------
    train_index : array-like
        The indices of the samples in the training set.
    test_index : array-like
        The indices of the samples in the test set.
    X : pandas DataFrame
        The features of the dataset.
    y : pandas Series
        The target variable of the dataset.
    num_feature2_use : int
        The number of features to select.
    cvncvsel : str
        Whether to use the config for nested cv or recursive nested cv.
    
    Returns
    -------
    X_train_selected : pandas DataFrame
        The filtered training set.
    X_test_selected : pandas DataFrame
        The filtered test set.
    num_feature : int or str
        The number of features selected or "full" if all features were selected.
    """
    
    data_loader = DataLoader(label=config["dataset_label"], csv_dir=config["dataset_name"])
    
    # Find the train and test sets
    X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
    y_train, _ = y[train_index], y[test_index]

    X_train, X_test = data_loader.normalize(
        X=X_tr,
        train_test_set=True,
        X_test=X_te,
        method=config["normalization"],
    )

    # Manipulate the missing values for both train and test sets
    X_train = data_loader.missing_values(
        data=X_train, method=config["missing_values_method"]
    )
    X_test = data_loader.missing_values(
        data=X_test, method=config["missing_values_method"]
    )

    # Find the feature selection type and apply it to the train and test sets
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
                num_feature = "full"
            else:
                raise ValueError(
                    "num_features must be an integer less than the number of features in the dataset"
                )
    elif config["feature_selection_type"] == "percentile":
        if isinstance(num_feature2_use, int):
            if (
                num_feature2_use == 100 or num_feature2_use == X.shape[1]
            ):  
                X_train_selected = X_train
                X_test_selected = X_test
                num_feature = "full"
            elif num_feature2_use < 100 and num_feature2_use > 0:
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
                    "num_features must be an integer less or equal than 100 and hugher thatn 0"
                )
    else:
        raise ValueError("num_features must be an integer or a list or None")

    return X_train_selected, X_test_selected, num_feature