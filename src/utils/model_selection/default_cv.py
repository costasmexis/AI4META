import time
from sklearn.model_selection import StratifiedKFold
import progressbar
from src.utils.model_selection.train import _fit_procedure

def _cv_loop(X, y, config, i, avail_thr):
    """
    Perform a cross-validation loop with default hyperparameters.

    This function executes cross-validation for fast evaluation of estimators' performance,
    focusing on obtaining initial insights for large datasets.

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
        Number of available threads for parallelization.

    Returns:
    --------
    list
        A list of dictionaries, each containing the results for a split in the cross-validation.

    Notes:
    ------
    - Results are generated using default hyperparameters for rapid assessment.
    - Progress bar indicates the status of the cross-validation process.
    """
    start = time.time()  # Start timer for the loop

    # Configure cross-validation splits
    config["cv_splits"] = StratifiedKFold(
        n_splits=config["splits"], shuffle=True, random_state=i
    )

    train_test_indices = list(config["cv_splits"].split(X, y))

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
        prefix=f"Round {i+1} of CV:",
        max_value=config["splits"],
        widgets=widgets,
    ) as bar:
        for split_index, (train_index, test_index) in enumerate(train_test_indices):
            # Initialize results dictionary
            results = {
                "Classifiers": [],
                "Selected_Features": [],
                "Number_of_Features": [],
                "Way_of_Selection": [],
                "Estimator": [],
                "Samples_counts": [],
            }
            results.update({f"{metric}": [] for metric in config["extra_metrics"]})

            # Perform fitting procedure
            results = _fit_procedure(X, y, config, results, train_index, test_index, i)
            temp_list.append([results])
            bar.update(split_index)
            time.sleep(0.5)

    # Flatten results
    list_dfs = [item for sublist in temp_list for item in sublist]

    # Log the duration of this round
    end = time.time()
    print(f"Finished with {i+1} round after {(end-start)/3600:.2f} hours.")

    return list_dfs
