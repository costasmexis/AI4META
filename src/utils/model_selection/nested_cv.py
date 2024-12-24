import time
from sklearn.model_selection import StratifiedKFold
import progressbar
from src.utils.model_selection.fit import _fit_procedure

def _inner_loop(X, y, config, train_index, test_index, avail_thr, i):
    """
    This function is used to perform the inner loop of the nested cross-validation for the selection of the best hyperparameters.
    Note: Return a list because this is the desired output for the parallel loop
    Runs only on the nested cross-validation function.
    Supports several inner selection methods.
    """
    opt_grid = "NestedCV"
    
    if config["parallel"] == "thread_per_round":
        n_jobs = 1
    elif config["parallel"] == "freely_parallel":
        n_jobs = avail_thr

    # Initialize variables
    results = {
        "Classifiers": [],
        "Selected_Features": [],
        "Number_of_Features": [],
        "Hyperparameters": [],
        "Way_of_Selection": [],
        "Estimator": [],
        'Samples_counts': [],
        'Inner_selection_mthd': [],
    }
    results.update({f"{metric}": [] for metric in config["extra_metrics"]})
    
    # Start fitting 
    results = _fit_procedure(X, y, config, results, train_index, test_index, i, n_jobs)
                    
    # Check for consistent list lengths
    lengths = {key: len(val) for key, val in results.items()}

    if len(set(lengths.values())) > 1:
        print("Inconsistent lengths in results:", lengths)
        raise ValueError("Inconsistent lengths in results dictionary")

    return [results]

def _outer_loop(X, y, config, i, avail_thr):
    """
    This function is used to make the separations of train and test data of the outer cross validation and initiates the parallilization
    with respect to the parallelization method.
    Uses different random seed for each round of the outer cross validation
    """
    start = time.time()  # Count time of outer loops

    # Split the data into train and test
    config["inner_cv"] = StratifiedKFold(
        n_splits=config["inner_splits"], shuffle=True, random_state=i
    )
    config["outer_cv"] = StratifiedKFold(
        n_splits=config["outer_splits"], shuffle=True, random_state=i
    )

    train_test_indices = list(config["outer_cv"].split(X, y))

    # Store the results in a list od dataframes
    list_dfs = []

    # Initiate the progress bar
    widgets = [
        progressbar.Percentage(),
        " ",
        progressbar.GranularBar(),
        " ",
        progressbar.Timer(),
        " ",
        progressbar.ETA(),
    ]
    
    # Find the parallelization method
    if config["parallel"] == "freely_parallel":
        temp_list = []
        with progressbar.ProgressBar(
            prefix=f"Outer fold of {i+1} round:",
            max_value=config["outer_splits"],
            widgets=widgets,
        ) as bar:
            split_index = 0

            # For each outer fold perform the inner loop
            for train_index, test_index in train_test_indices:
                results = _inner_loop(
                    X, y, config, train_index, test_index, avail_thr, i
                )
                temp_list.append(results)
                bar.update(split_index)
                split_index += 1
                time.sleep(1)
            list_dfs = [item for sublist in temp_list for item in sublist]
            end = time.time()
    else:
        temp_list = []
        with progressbar.ProgressBar(
            prefix=f"Outer fold of {i+1} round:",
            max_value=config["outer_splits"],
            widgets=widgets,
        ) as bar:
            split_index = 0

            # For each outer fold perform the inner loop
            for train_index, test_index in train_test_indices:
                results = _inner_loop(
                    X, y, config, train_index, test_index, avail_thr, i
                )
                temp_list.append(results)
                bar.update(split_index)
                split_index += 1
                time.sleep(1)
            list_dfs = [item for sublist in temp_list for item in sublist]
            end = time.time()

    # Return the list of dataframes and the time of the outer loop
    print(f"Finished with {i+1} round after {(end-start)/3600:.2f} hours.")
    return list_dfs