import time
from sklearn.model_selection import StratifiedKFold
import progressbar
from .fit import _fit_procedure

def _cv_loop(X, y, config, i, avail_thr):
    """
    This function is used to perform the rounds loop of the cross-validation using only the default parameters. 
    It is useful for first insights about the estimators performance and is fast for big datasets.
    It is used only in rcv_accel function.

    Parameters
    ----------
    i : int
        The current round of the cross-validation.

    avail_thr : int
        The number of available threads for parallelization.

    Returns
    -------
    list_dfs : list of pandas DataFrames
        A list of dataframes containing the results of the cross-validation.
    """
    start = time.time()  # Count time of outer loops
    
    # Split the data into train and test
    config["cv_splits"] = StratifiedKFold(
        n_splits=config["splits"], shuffle=True, random_state=i
    )

    train_test_indices = list(config["cv_splits"].split(X, y))

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

    temp_list = []
    with progressbar.ProgressBar(
        prefix=f"Round {i+1} of CV:",
        max_value=config["splits"],
        widgets=widgets,
    ) as bar:
        split_index = 0

        # For each outer fold perform
        for train_index, test_index in train_test_indices:
            # Initialize variables
            results = {
                "Classifiers": [],
                "Selected_Features": [],
                "Number_of_Features": [],
                "Way_of_Selection": [],
                "Estimator": [],
                'Samples_counts': [],
            }
            results.update({f"{metric}": [] for metric in config["extra_metrics"]})
            
            results = _fit_procedure(X, y, config, results, train_index, test_index, i)
            temp_list.append([results])
            bar.update(split_index)
            split_index += 1
            time.sleep(0.5)

        list_dfs = [item for sublist in temp_list for item in sublist]
        end = time.time()

    # Return the list of dataframes and the time fold
    print(f"Finished with {i+1} round after {(end-start)/3600:.2f} hours.")
    return list_dfs