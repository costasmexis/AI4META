import time
from sklearn.model_selection import StratifiedKFold
import progressbar
from src.utils.model_selection.train import _fit_procedure

def _inner_loop(X, y, config, train_index, test_index, avail_thr, i):
    """
    Perform the inner loop of nested cross-validation.

    This function handles the training and validation within each split of the inner cross-validation.
    It evaluates models and hyperparameters using the specified configuration.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature dataset.
    y : pandas.Series or array-like
        Target labels.
    config : dict
        Configuration dictionary containing hyperparameters and other settings.
    train_index : array-like
        Indices for the training set.
    test_index : array-like
        Indices for the testing set.
    avail_thr : int
        Number of threads available for parallelization.
    i : int
        Current iteration number for reproducibility.

    """
    # Determine the number of jobs for parallel execution
    n_jobs = 1 if config["parallel"] == "thread_per_round" else avail_thr

    # Initialize results dictionary
    results = {
        "Classifiers": [],
        "Selected_Features": [],
        "Number_of_Features": [],
        "Hyperparameters": [],
        "Way_of_Selection": [],
        "Estimator": [],
        "Samples_counts": [],
        "Inner_selection_mthd": [],
    }
    results.update({f"{metric}": [] for metric in config["extra_metrics"]})

    # Execute the fitting procedure
    results = _fit_procedure(X, y, config, results, train_index, test_index, i, n_jobs)

    # # Validate consistency of results lengths
    # lengths = {key: len(val) for key, val in results.items()}

    # if len(set(lengths.values())) > 1:
    #     print("Inconsistent lengths in results:", lengths)
    #     raise ValueError("Inconsistent lengths in results dictionary")

    return [results]

def _outer_loop(X, y, config, i, avail_thr):
    """
    Perform the outer loop of nested cross-validation.

    This function manages the outer cross-validation splits and calls the inner loop
    for model evaluation and hyperparameter selection.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature dataset.
    y : pandas.Series or array-like
        Target labels.
    config : dict
        Configuration dictionary containing hyperparameters and other settings.
    i : int
        Current round number for reproducibility.
    avail_thr : int
        Number of threads available for parallelization.
    """
    start = time.time()  # Track the start time for this round

    # Configure inner and outer cross-validation
    config["inner_cv"] = StratifiedKFold(
        n_splits=config["inner_splits"], shuffle=True, random_state=i
    )
    config["outer_cv"] = StratifiedKFold(
        n_splits=config["outer_splits"], shuffle=True, random_state=i
    )

    train_test_indices = list(config["outer_cv"].split(X, y))

    # Initialize progress bar
    widgets = [
        progressbar.Percentage(),
        " ",
        progressbar.GranularBar(),
        " ",
        progressbar.Timer(),
        " ",
        progressbar.ETA(),
    ]

    # Store results from all splits
    temp_list = []
    with progressbar.ProgressBar(
        prefix=f"Outer fold of {i+1} round:",
        max_value=config["outer_splits"],
        widgets=widgets,
    ) as bar:
        for split_index, (train_index, test_index) in enumerate(train_test_indices):
            # Call the inner loop for each split
            results = _inner_loop(X, y, config, train_index, test_index, avail_thr, i)
            temp_list.append(results)
            bar.update(split_index)
            time.sleep(1)

    list_dfs = [item for sublist in temp_list for item in sublist]  # Flatten results
    end = time.time()

    # Log the duration of this round
    print(f"Finished with {i+1} round after {(end-start)/3600:.2f} hours.")

    return list_dfs
