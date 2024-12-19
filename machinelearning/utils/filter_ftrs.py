from dataloader import DataLoader
from sklearn.feature_selection import SelectFromModel

def _filter_features(X, y, num_feature2_use, config, train_index = None, test_index = None, features_names_list = None):
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
        The number of features selected or "none" if all features were selected.
    """
    
    data_loader = DataLoader(label=config["dataset_label"], csv_dir=config["dataset_name"])
    
    if (train_index is None) and (test_index is None):
        X_tr, X_te = X.copy(), X.copy()
        y_train = y.copy()
    else:
        # Find the train and test sets
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y[train_index], y[test_index]

    if features_names_list is not None:
        X_tr = X_tr[features_names_list]
        X_te = X_te[features_names_list]

    # Normalize the train and test sets
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
                num_feature = "none"
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
                num_feature = "none"
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

def _sfm(estimator, X_train, X_test, y_train, num_feature2_use=None, threshold="mean"):
    """
    Select features using SelectFromModel with either a predefined number of features 
    or using the threshold if num_feature2_use is not provided.

    Args:
        estimator: The estimator object to use for feature selection. Must support `fit` and `feature_importances_`.
        X_train: Training feature set.
        X_test: Testing feature set.
        y_train: Training target variable.
        num_feature2_use: Number of features to select, defaults to None.
        threshold: Threshold value for feature selection, defaults to "mean".

    Returns:
        X_train_selected: The training set with selected features.
        X_test_selected: The testing set with selected features.
        num_feature2_use: The number of features selected.
    """
    
    # Fit the estimator on the training data
    estimator.fit(X_train, y_train)

    # Initialize SelectFromModel based on num_feature2_use or threshold
    if num_feature2_use is None:
        sfm = SelectFromModel(estimator, threshold=threshold)
    else:
        sfm = SelectFromModel(estimator, max_features=num_feature2_use)

    # Fit SelectFromModel to identify important features
    sfm.fit(X_train, y_train)
    
    # Get indices of selected features
    selected_features = sfm.get_support(indices=True)
    selected_columns = X_train.columns[selected_features].to_list()
    
    # Select the features for training and testing sets
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]

    return X_train_selected, X_test_selected, num_feature2_use