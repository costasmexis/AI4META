import numpy as np

def _calc_ci_btstrp(data, central_tendency='median', n_bootstraps=1000, confidence_level=0.95):
    """
    Calculate the confidence interval of the mean or median using bootstrapping.

    Parameters:
    -----------
    data : array-like
        Input data to calculate the confidence interval for.
    central_tendency : str, optional
        Type of central tendency to compute ('mean' or 'median'). Defaults to 'median'.
    n_bootstraps : int, optional
        Number of bootstrap samples to generate. Defaults to 1000.
    confidence_level : float, optional
        Confidence level for the interval. Defaults to 0.95 (95%).

    Returns:
    --------
    tuple
        Lower and upper bounds of the confidence interval.

    Raises:
    -------
    ValueError
        If `central_tendency` is not one of 'mean' or 'median'.
    """
    if central_tendency not in ['mean', 'median']:
        raise ValueError("Invalid central_tendency value. Choose 'mean' or 'median'.")

    ms = []
    for _ in range(n_bootstraps):
        # Generate a bootstrap sample
        sample = np.random.choice(data, size=len(data), replace=True)

        # Compute the desired central tendency
        if central_tendency == 'median':
            ms.append(np.median(sample))
        elif central_tendency == 'mean':
            ms.append(np.mean(sample))

    # Calculate the lower and upper bounds of the confidence interval
    alpha = 1 - confidence_level
    lower_bound = np.percentile(ms, alpha / 2 * 100)
    upper_bound = np.percentile(ms, (1 - alpha / 2) * 100)

    return lower_bound, upper_bound
