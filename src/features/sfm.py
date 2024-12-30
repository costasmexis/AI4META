from sklearn.feature_selection import SelectFromModel
import numpy as np

def _sfm(estimator, X_train, X_test, y_train, num_feature2_use=None, threshold="mean"):
    """
    Perform feature selection using SelectFromModel.

    This function selects features based on the importance weights provided by a fitted
    estimator. The selection can be performed either by specifying a threshold or limiting
    the number of features to be selected.

    Parameters:
    -----------
    estimator : object
        The estimator object to use for feature selection. Must support `fit` and `feature_importances_`.
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature set.
    X_test : pandas.DataFrame or numpy.ndarray
        Testing feature set.
    y_train : pandas.Series or numpy.ndarray
        Training target variable.
    num_feature2_use : int or None, optional
        Number of features to select. If None, `threshold` is used to determine the selection criteria.
        Defaults to None.
    threshold : str or float, optional
        Threshold value for feature selection. Can be 'mean', 'median', or a numeric value.
        Defaults to 'mean'.

    Returns:
    --------
    X_train_selected : pandas.DataFrame or numpy.ndarray
        Training set with selected features.
    X_test_selected : pandas.DataFrame or numpy.ndarray
        Testing set with selected features.
    num_feature2_use : int
        The number of features selected.

    Notes:
    ------
    - The estimator must be fitted with the training data before feature selection.
    - The method dynamically adapts to either use `threshold` or `max_features` based on `num_feature2_use`.
    """
    # Fit the estimator on the training data
    estimator.fit(X_train, y_train)

    # Initialize SelectFromModel
    if num_feature2_use is None:
        sfm = SelectFromModel(estimator, threshold=threshold)
    else:
        sfm = SelectFromModel(estimator, max_features=num_feature2_use, threshold=-np.inf)

    # Fit SelectFromModel to identify important features
    sfm.fit(X_train, y_train)

    # Get indices and names of selected features
    selected_features = sfm.get_support(indices=True)
    selected_columns = X_train.columns[selected_features].to_list()

    # Select the features for training and testing sets
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]

    # Update the number of features selected if it was None
    if num_feature2_use is None:
        num_feature2_use = len(selected_columns)

    return X_train_selected, X_test_selected, num_feature2_use