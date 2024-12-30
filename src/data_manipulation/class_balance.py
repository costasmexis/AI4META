from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks

def _class_balance(X, y, bal_type, i=42):
    """
    Apply class balancing strategies to address imbalanced datasets.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature dataset.
    y : pandas.Series or numpy.ndarray
        Target labels.
    bal_type : dict or None
        Dictionary specifying the class balancing method. Options include:
        - 'smote': Synthetic Minority Over-sampling Technique.
        - 'borderline_smote': Borderline SMOTE for over-sampling.
        - 'tomek': Tomek links for under-sampling.
        If None, no class balancing is applied.
    i : int, optional
        Random state seed for reproducibility. Defaults to 42.

    Returns:
    --------
    tuple
        X_balanced : pandas.DataFrame or numpy.ndarray
            Balanced feature dataset.
        y_balanced : pandas.Series or numpy.ndarray
            Balanced target labels.

    Raises:
    -------
    ValueError
        If an unsupported class balancing method is specified.
    """
    # No class balancing applied
    if bal_type is None:
        return X, y

    # Apply class balancing based on the specified method
    if bal_type['class_balance'] == 'smote':
        X_balanced, y_balanced = SMOTE(random_state=i).fit_resample(X, y)
    elif bal_type['class_balance'] == 'borderline_smote':
        X_balanced, y_balanced = BorderlineSMOTE(random_state=i).fit_resample(X, y)
    elif bal_type['class_balance'] == 'tomek':
        tomek = TomekLinks()
        X_balanced, y_balanced = tomek.fit_resample(X, y)
    else:
        raise ValueError(
            f"Unsupported class balancing method: {bal_type['class_balance']}."
            " Choose from ['smote', 'borderline_smote', 'tomek'] or None."
        )

    return X_balanced, y_balanced
