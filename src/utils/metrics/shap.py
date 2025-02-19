from typing import Any, Union, Optional
import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray

def _calc_shap(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    model: Any
) -> NDArray:
    """
    Calculate SHAP (SHapley Additive exPlanations) values for model predictions.
    
    This function computes feature importance values that show how each feature 
    contributes to model predictions. It handles different model types and adjusts
    the calculation approach based on model characteristics.

    Parameters
    ----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Training dataset used to initialize the SHAP explainer.
        This provides the baseline for feature distributions.
    X_test : Union[pd.DataFrame, np.ndarray]
        Test dataset for which SHAP values will be calculated.
        These are the actual instances we want to explain.
    model : Any
        Trained model for which SHAP values will be computed.
        Must implement either a prediction method or predict_proba.

    Returns
    -------
    np.ndarray
        Array of SHAP values for the test dataset.
        Shape: (n_samples, n_features) for binary classification
        or (n_samples, n_features, n_classes) for multiclass.

    Raises
    ------
    TypeError
        If the model is not compatible with SHAP's explainer.
    ValueError
        If there are issues with the dimensionality or format of input data.

    Notes
    -----
    - For tree-based models (like LGBMClassifier, CatBoostClassifier, 
      RandomForestClassifier), the function uses specialized handling with 
      check_additivity=False for better performance.
    - For models that don't directly support SHAP's callable interface,
      the function falls back to using predict_proba.
    - The function automatically adjusts the number of evaluations based on
      the feature dimensionality if needed.
    """
    try:
        # Initialize SHAP explainer
        explainer = shap.Explainer(model, X_train)
    except TypeError as e:
        # Handle models that need predict_proba wrapper
        if "The passed model is not callable" in str(e):
            explainer = shap.Explainer(
                lambda x: model.predict_proba(x), 
                X_train
            )
        else:
            raise TypeError(
                f"Model {type(model).__name__} is not compatible with SHAP: {str(e)}"
            )

    try:
        # Calculate SHAP values with model-specific handling
        if model.__class__.__name__ in [
            "LGBMClassifier",
            "CatBoostClassifier",
            "RandomForestClassifier"
        ]:
            # Tree-based models need special handling
            shap_values = explainer(X_test, check_additivity=False)
        else:
            # Standard SHAP calculation
            shap_values = explainer(X_test)

    except ValueError as e:
        # Handle cases requiring adjusted evaluation count
        num_features = X_test.shape[1]
        max_evals = 2 * num_features + 1
        
        try:
            shap_values = explainer(X_test, max_evals=max_evals)
        except Exception as nested_e:
            raise ValueError(
                f"Failed to calculate SHAP values even with adjusted evaluations: {str(nested_e)}"
            )

    # Handle multi-dimensional SHAP values for classification
    if len(shap_values.shape) == 3:
        # For binary classification, return values for positive class
        shap_values = shap_values[:, :, 1]

    return shap_values.values if hasattr(shap_values, 'values') else shap_values