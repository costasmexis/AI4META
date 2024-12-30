import shap

def _calc_shap(X_train, X_test, model):
    """
    Calculate SHAP values for a given model and dataset.

    This function uses SHAP (SHapley Additive exPlanations) to compute the
    feature importance values for predictions made by the provided model.

    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training dataset used to initialize the SHAP explainer.
    X_test : pandas.DataFrame or numpy.ndarray
        Test dataset for which SHAP values are calculated.
    model : object
        Trained model for which SHAP values are to be computed.

    Returns:
    --------
    numpy.ndarray
        Calculated SHAP values for the test dataset.

    Notes:
    ------
    - Supports models like `LGBMClassifier`, `CatBoostClassifier`, and
      `RandomForestClassifier` with specialized handling for certain scenarios.
    - Handles compatibility issues with models that do not directly support SHAP.
    - Dynamically adjusts SHAP calculation settings in case of errors.
    """
    try:
        # Initialize SHAP explainer
        explainer = shap.Explainer(model, X_train)
    except TypeError as e:
        # Handle compatibility issues for models like predict_proba
        if "The passed model is not callable" in str(e):
            print("Switching to predict_proba due to compatibility issue with the model.")
            explainer = shap.Explainer(lambda x: model.predict_proba(x), X_train)
        else:
            raise TypeError(e)

    try:
        # Compute SHAP values with specialized handling for certain models
        if model.__class__.__name__ in ["LGBMClassifier", "CatBoostClassifier", "RandomForestClassifier"]:
            shap_values = explainer(X_test, check_additivity=False)
        else:
            shap_values = explainer(X_test)
    except ValueError:
        # Adjust max evaluations dynamically if error occurs
        num_features = X_test.shape[1]
        max_evals = 2 * num_features + 1
        shap_values = explainer(X_test, max_evals=max_evals)

    # Handle multi-dimensional SHAP values
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]  # Select class 1 values for classification

    return shap_values
