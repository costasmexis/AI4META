from typing import Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import logging
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks

# Global logging flags for class balancing
_logged_balance_methods = {}

def _log_once(logger, method: str, message: str) -> None:
    """Log a message only once for a specific balancing method."""
    if method not in _logged_balance_methods:
        _logged_balance_methods[method] = True
        logger.info(message)

def _class_balance(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    balance_method: str = None,
    i: int = 42
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Apply class balancing strategies to address imbalanced datasets.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature dataset.
    y : Series or ndarray
        Target labels.
    bal_type : dict or None
        Class balancing configuration. Format: {'class_balance': method}
        Supported methods:
        - 'smote': Synthetic Minority Over-sampling Technique
        - 'borderline_smote': Borderline SMOTE for over-sampling
        - 'tomek': Tomek links for under-sampling
        If None, no balancing is applied.
    i : int
        Random state seed for reproducibility.

    Returns
    -------
    X_balanced : DataFrame or ndarray
        Balanced feature dataset.
    y_balanced : Series or ndarray
        Balanced target labels.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)

    # No class balancing if bal_type is None
    if balance_method is None:
        _log_once(logger, 'none', "✓ No class balancing requested, returning original data")
        return X, y
    
    try:
        if balance_method == 'smote':
            balancer = SMOTE(random_state=i)
            _log_once(logger, 'smote', "Applying SMOTE oversampling...")
        elif balance_method == 'borderline_smote':
            balancer = BorderlineSMOTE(random_state=i)
            _log_once(logger, 'borderline_smote', "Applying Borderline SMOTE oversampling...")
        elif balance_method == 'tomek':
            balancer = TomekLinks()
            _log_once(logger, 'tomek', "Applying Tomek links undersampling...")
        else:
            raise ValueError(
                f"Unsupported balancing method: {balance_method}. "
                "Choose from ['smote', 'borderline_smote', 'tomek']"
            )

        # Perform resampling
        X_balanced, y_balanced = balancer.fit_resample(X, y)

        # # Log final class distribution
        # unique, counts = np.unique(y_balanced, return_counts=True)
        # final_dist = dict(zip(unique, counts))
        # logger.info(f"Final class distribution: {final_dist}")

        # # Calculate and log the changes
        # samples_diff = len(y_balanced) - len(y)
        # logger.info(f"✓ Balancing complete: {abs(samples_diff)} samples " + 
        #            ("added" if samples_diff > 0 else "removed"))

        return X_balanced, y_balanced

    except Exception as e:
        logger.error(f"Error during class balancing: {str(e)}")
        raise