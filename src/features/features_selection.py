from src.data.dataloader import DataLoader
from typing import Optional, Union, List, Tuple, Dict
import pandas as pd
import numpy as np
import logging

# Global logging flags for feature selection operations
_logged_operations = {}

def _log_once(logger, operation: str, message: str) -> None:
    """Log a message only once for a specific operation."""
    if operation not in _logged_operations:
        _logged_operations[operation] = True
        logger.info(message)

def preprocess(
    config: Dict,
    num_features: int,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[pd.DataFrame] = None,
    features_names_list: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Union[List[str], str]]:
    """
    Select features using configured method with support for train/test splits.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing preprocessing settings
    num_features : int
        Number of features to select
    X_train : DataFrame
        Training feature matrix
    y_train : Series or ndarray
        Training target variable
    X_test : DataFrame, optional
        Test feature matrix
    data_loader : DataLoader, optional
        Existing DataLoader instance to use
        
    Returns
    -------
    DataFrame
        Training data with selected features
    DataFrame or None
        Test data with selected features (if X_test provided)
    List[str] or str
        Names of selected features or 'none' if no selection performed
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Validate DataLoader
    loader = DataLoader(csv_dir=config.get('csv_dir'), label=config.get('label'))
    if loader is None:
        raise ValueError("No DataLoader instance available")
    
    # Check for feature list
    if features_names_list is not None:
        X_train = X_train[features_names_list]
        if X_test is not None:
            X_test = X_test[features_names_list]
        _log_once(logger, 'feature_list_complete', 
                 f"✓ Selected {len(features_names_list)} features from list")
        
    # Normalize data
    if X_test is not None:
        X_train_norm, X_test_norm = loader.normalize(
            X=X_train,
            method=config.get('normalization'),
            X_test=X_test
        )
    else:
        X_train_norm = loader.normalize(
            X=X_train,
            method=config.get('normalization')
        )
        X_test_norm = None

    # Handle missing values
    X_train_clean = loader.missing_values(X_train_norm, method=config.get('missing_values'))
    if X_test_norm is not None:
        X_test_clean = loader.missing_values(X_test_norm, method=config.get('missing_values'))
    else:
        X_test_clean = None

    # Check if feature selection is needed
    if ((num_features == X_train_clean.shape[1]) or 
        (num_features is None) or 
        ((config.get('feature_selection_type') == 'percentile') and (num_features == 100))):
        _log_once(logger, 'no_selection',
                 "✓ No feature selection needed - using all features")
        return X_train_clean, X_test_clean, 'none'
    
    # Validate number of features
    if (num_features > X_train_clean.shape[1]):
        raise ValueError(
            f"Number of features to select ({num_features}) is greater "
            f"than the number of features in the dataset ({X_train_clean.shape[1]})"
        )

    # Perform feature selection
    selected_features = loader.feature_selection(
        X=X_train_clean,
        y=y_train,
        method=config.get('feature_selection_type'),
        num_features=num_features,
        inner_method=config.get('feature_selection_method')
    )

    # Apply feature selection
    X_train_selected = X_train_clean[selected_features]
    X_test_selected = X_test_clean[selected_features] if X_test_clean is not None else None
    return X_train_selected, X_test_selected, num_features