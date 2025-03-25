import time
from typing import List, Dict, Any
import numpy as np
import progressbar
from sklearn.model_selection import StratifiedKFold
from src.utils.model_selection.train import _fit_procedure

def _cv_loop(
    X: Any,
    y: np.ndarray,
    config: Dict[str, Any],
    round_num: int,
    avail_thr: int
) -> List[Dict[str, Any]]:
    """
    Execute a cross-validation loop with default hyperparameters for model evaluation.

    This function performs stratified k-fold cross-validation to evaluate models using
    their default hyperparameters. It's designed for rapid initial assessment of model
    performance, particularly useful for large datasets or when quick insights are needed.

    Parameters
    ----------
    X : array-like
        Feature matrix of shape (n_samples, n_features). Can be pandas DataFrame
        or numpy array.
    y : numpy.ndarray
        Target values of shape (n_samples,).
    config : dict
        Configuration dictionary containing:
        - 'splits': int, number of cross-validation folds
        - 'extra_metrics': list, metrics to evaluate
        - Other model-specific parameters and feature selection settings
    round_num : int
        Current round number (used for reproducibility and progress tracking).
    avail_thr : int
        Number of available threads for parallel processing.

    """
    # Record start time for performance monitoring
    start_time = time.time()

    # Initialize stratified k-fold cross-validation
    cv_splitter = StratifiedKFold(
        n_splits=config["splits"],
        shuffle=True,
        random_state=round_num
    )
    train_test_indices = list(cv_splitter.split(X, y))

    # Configure progress bar for visual feedback
    widgets = [
        progressbar.Percentage(),
        " ",
        progressbar.GranularBar(),
        " ",
        progressbar.Timer(),
        " ",
        progressbar.ETA(),
    ]

    # Initialize list to store results from all folds
    fold_results = []

    # Execute cross-validation with progress tracking
    with progressbar.ProgressBar(
        prefix=f"Round {round_num + 1} of CV:",
        max_value=config["splits"],
        widgets=widgets,
    ) as progress_bar:
        
        # Iterate through each fold
        for fold_idx, (train_index, test_index) in enumerate(train_test_indices):
            # Initialize results dictionary for current fold
            fold_metrics = {
                "Classifiers": [],
                "Selected_Features": [],
                "Number_of_Features": [],
                "Way_of_Selection": [],
                "Estimator": [],
                "Samples_counts": [],
            }
            
            # Add additional metric tracking
            fold_metrics.update({
                metric: [] for metric in config["extra_metrics"]
            })

            # Perform model fitting and evaluation for current fold
            fold_metrics = _fit_procedure(
                X=X,
                y=y,
                config=config,
                results=fold_metrics,
                train_index=train_index,
                test_index=test_index,
                i=round_num
            )
            
            # Store results from current fold
            fold_results.append([fold_metrics])
            
            # Update progress bar
            progress_bar.update(fold_idx)
            time.sleep(0.5)  # Short delay for progress bar visibility

    # Flatten results list for consistent format
    flattened_results = [
        item for sublist in fold_results for item in sublist
    ]

    # Calculate and log execution time
    end_time = time.time()
    execution_hours = (end_time - start_time) / 3600
    print(f"Completed round {round_num + 1} in {execution_hours:.2f} hours")

    return flattened_results
