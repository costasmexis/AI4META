from sklearn.feature_selection import SelectFromModel
import numpy as np
import logging
from typing import Tuple, Optional, Union
import pandas as pd

# Global logging flags for SFM operations
_logged_operations = {}

def _log_once(logger, operation: str, message: str) -> None:
    """Log a message only once for a specific operation."""
    if operation not in _logged_operations:
        _logged_operations[operation] = True
        logger.info(message)

def _sfm(
    estimator: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
     num_features: Optional[int] = None,
    threshold: Union[str, float, int] = "mean"
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Perform feature selection using SelectFromModel.

    Parameters
    ----------
    estimator : object
        The estimator object to use for feature selection.
    X_train : DataFrame
        Training feature set.
    X_test : DataFrame
        Testing feature set.
    y_train : Series or ndarray
        Training target variable.
     num_features : int or None, optional
        Number of features to select. If None, threshold is used.
    threshold : str or float, optional
        Threshold value for feature selection ('mean', 'median', or numeric).

    Returns
    -------
    X_train_selected : DataFrame
        Training set with selected features.
    X_test_selected : DataFrame
        Testing set with selected features.
     num_features : int
        Number of features selected.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)

    # Fit the estimator
    estimator.fit(X_train, y_train)
    if  num_features is None:
        sfm = SelectFromModel(estimator, threshold=threshold)
    else:
        sfm = SelectFromModel(estimator, max_features= num_features, threshold=-np.inf)

    # Fit SelectFromModel
    sfm.fit(X_train, y_train)

    # Get selected features
    selected_features = sfm.get_support(indices=True)
    selected_columns = X_train.columns[selected_features].to_list()

    # Select features
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]

    # Update number of features if using threshold
    if  num_features is None:
         num_features = len(selected_columns)

    # Log completion
    _log_once(logger, f'complete_{ num_features}',
             f"✓ Selected { num_features} features using SelectFromModel")

    return X_train_selected, X_test_selected,  num_features