import numpy as np
from scipy.stats import sem

def _calc_ci_btstrp(data, type='median'):
    """
    Calculate the confidence interval of the mean or median using bootstrapping.

    Args:
        data (array-like): Input data to calculate the confidence interval for.
        type (str): Type of central tendency to compute ('mean' or 'median'). Defaults to 'median'.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    ms = []
    for _ in range(1000):
        # Generate a bootstrap sample
        sample = np.random.choice(data, size=len(data), replace=True)
        
        # Compute the desired central tendency
        if type == 'median':
            ms.append(np.median(sample))
        elif type == 'mean':
            ms.append(np.mean(sample))
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = np.percentile(ms, (1 - 0.95) / 2 * 100)
    upper_bound = np.percentile(ms, (1 + 0.95) / 2 * 100)
    
    return lower_bound, upper_bound