from sklearn.feature_selection import SelectFromModel

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