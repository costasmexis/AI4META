import optuna
import numpy as np
from src.constants.parameters_grid import hyper_compl

def _gso_model(trials, model_name, splits, method):
    """
    Select the 'balanced trained' hyperparameters for a given model using gap score optimization.

    This method evaluates trials based on either training or validation scores and selects
    the hyperparameters with the smallest gap between test and train scores above a specified threshold.

    Parameters:
    -----------
    trials : list of optuna.trial.Trial
        List of completed trials to evaluate.
    model_name : str
        Name of the model for which hyperparameters are selected.
    splits : int
        Number of cross-validation splits.
    method : str
        Selection method: 'gso_1' or 'gso_2'.

    Returns:
    --------
    dict
        Selected hyperparameters.

    Raises:
    -------
    ValueError
        If the method is not 'gso_1' or 'gso_2'.
    """
    # Extract relevant trial data
    trials_data = [
        {
            "params": t.params,
            "mean_test_score": t.user_attrs.get("mean_test_score"),
            "mean_train_score": t.user_attrs.get("mean_train_score"),
            "gap_scores": [
                abs(t.user_attrs.get(f"split{i}_test_score") - t.user_attrs.get(f"split{i}_train_score"))
                for i in range(splits)
            ],
        }
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if method not in ["gso_1", "gso_2"]:
        raise ValueError(f"Unsupported method: {method}. Use 'gso_1' or 'gso_2'.")

    # Sort and filter based on the specified method
    key_score = "mean_train_score" if method == "gso_1" else "mean_test_score"
    trials_data.sort(key=lambda x: x[key_score], reverse=True)
    best_score = trials_data[0][key_score]
    score_threshold = 0.85 * best_score

    filtered_trials = [t for t in trials_data if t[key_score] >= score_threshold]

    # Select the trial with the smallest average gap score
    if filtered_trials:
        best_trial = min(filtered_trials, key=lambda x: np.mean(x["gap_scores"]))
        return best_trial["params"]

    return trials[0].params  # Default to the first trial if no filtered trials

def _one_sem_model(trials, model_name, samples, splits, method):
    """
    Select the 'simplest' hyperparameters using the one-standard-error rule.

    This method identifies hyperparameters with validation scores above a threshold
    (SEM-adjusted) and selects the simplest configuration based on model complexity.

    Parameters:
    -----------
    trials : list of optuna.trial.Trial
        List of completed trials to evaluate.
    model_name : str
        Name of the model for which hyperparameters are selected.
    samples : int
        Number of samples in the dataset.
    splits : int
        Number of cross-validation splits.
    method : str
        Selection method: 'one_sem' or 'one_sem_grd'.

    Returns:
    --------
    dict
        Selected hyperparameters.

    Raises:
    -------
    ValueError
        If the method is not 'one_sem' or 'one_sem_grd'.
    """
    if method not in ["one_sem", "one_sem_grd"]:
        raise ValueError(f"Unsupported method: {method}. Use 'one_sem' or 'one_sem_grd'.")

    trials_data = [
        {
            "params": t.params,
            "value": t.values[0],
            "sem": t.user_attrs["std_test_score"] / (splits**0.5),
        }
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    trials_data.sort(key=lambda x: (x["value"], -x["sem"]), reverse=True)

    best_score = trials_data[0]["value"]
    sem_threshold = best_score - trials_data[0]["sem"]
    filtered_trials = [t for t in trials_data if t["value"] >= sem_threshold]

    if method == "one_sem":
        for trial in filtered_trials:
            trial["complexity"] = _calculate_complexity(trial, model_name, samples)
        simplest_trial = min(filtered_trials, key=lambda x: (x["complexity"]))
        return simplest_trial["params"]

    # For one_sem_grd, prioritize hyperparameters
    hyperparams = hyper_compl[model_name]
    for hyper, order in hyperparams.items():
        filtered_trials.sort(key=lambda x: x["params"][hyper], reverse=not order)

    return filtered_trials[0]["params"]

def _calculate_complexity(trial, model_name, samples):
    """
    Calculate the complexity of a model based on its hyperparameters.

    Parameters:
    -----------
    trial : dict
        Dictionary containing trial parameters.
    model_name : str
        Name of the model.
    samples : int
        Number of samples in the dataset.

    Returns:
    --------
    float
        Calculated complexity.
    """
    params = trial["params"]
    if model_name in ["RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier", "LGBMClassifier"]:
        max_depth = params.get("max_depth", float("inf"))
        leaf_estimator = params.get("min_samples_leaf", params.get("min_child_samples", 1))
        actual_depth = min(samples / leaf_estimator, max_depth)
        return params["n_estimators"] * (2 ** (actual_depth - 1))
    elif model_name == "CatBoostClassifier":
        max_depth = params.get("depth", float("inf"))
        actual_depth = min(samples / params.get("min_data_in_leaf", 1), max_depth)
        return params["n_estimators"] * (2 ** (actual_depth - 1))
    elif model_name in ["LogisticRegression", "ElasticNet"]:
        return params["C"] * params["max_iter"]
    elif model_name == "SVC":
        if params.get("kernel") == "poly":
            return params["C"] * params["degree"]
        return params["C"]
    elif model_name == "KNeighborsClassifier":
        return params["leaf_size"]
    elif model_name == "LinearDiscriminantAnalysis":
        return params.get("shrinkage", 1.0)
    elif model_name == "GaussianNB":
        return params.get("var_smoothing", 1.0)
    elif model_name == "GaussianProcessClassifier":
        return params.get("max_iter_predict", 1) * params.get("n_restarts_optimizer", 1)

    return float("inf")  # Fallback for unrecognized models




# def _gso_model(trials, model_name, splits, method):
#     """
#     This function selects the 'balanced trainned' hyperparameters for the given model.
    
#     The 'balanced trainned' hyperparameters are defined as the hyperparameters that have
#     the best train or validation score and a score threshold that is 85% of the
#     best score. The hyperparameters are selected by sorting the trials by score
#     and then filtering the trials by those that are above the score threshold.
#     The trial with the smallest average gap score is then selected.
    
#     Parameters
#     ----------
#     trials : list of optuna.trial.Trial
#         List of trials to select the best hyperparameters from
#     model_name : str
#         Name of the model to select the hyperparameters for
#     splits : int
#         Number of splits to calculate the gap score
#     method : str
#         Method to select the hyperparameters. Can be 'gso_1' or 'gso_2'
    
#     Returns
#     -------
#     dict
#         The selected hyperparameters as a dictionary
#     """
#     trials_data = [
#         {
#             "params": t.params,
#             "mean_test_score":  t.user_attrs.get('mean_test_score'),
#             "mean_train_score": t.user_attrs.get('mean_train_score'),
#             "test_scores": [t.user_attrs.get(f"split{i}_test_score") for i in range(splits)],
#             "train_scores": [t.user_attrs.get(f"split{i}_train_score") for i in range(splits)],
#             "gap_scores": [np.abs(t.user_attrs.get(f"split{i}_test_score") - t.user_attrs.get(f"split{i}_train_score")) for i in range(splits)]
#         }
#         for t in trials
#         if t.state == optuna.trial.TrialState.COMPLETE
#     ]
    
#     if method == "gso_1":
#         # Sort trials by mean train score
#         trials_data = sorted(
#                     trials_data, key=lambda x: (x["mean_train_score"]), reverse=True
#                 )

#         # Find the best mean train score and set a threshold
#         best_train_score = trials_data[0]["mean_train_score"]
#         k = 0.85
#         train_score_threshold = k * best_train_score

#         # Filter trials by those that are above the test score threshold and have a train score not lower than the test score
#         filtered_trials = [t for t in trials_data if (t["mean_train_score"] >= train_score_threshold)]

#         # Select the trial with the smallest average gap score
#         if filtered_trials:
#             gso_1_trial = min(filtered_trials, key=lambda x: np.mean(x["gap_scores"]))
#             return gso_1_trial["params"]
#         else:
#             return trials[0].params
        
#     elif method == "gso_2":
#         # Sort trials by mean test score
#         trials_data = sorted(
#             trials_data, key=lambda x: (x["mean_test_score"]), reverse=True
#         )

#         # Find the best mean validation score and set a threshold
#         best_test_score = trials_data[0]["mean_test_score"]
#         k = 0.85
#         test_score_threshold = k * best_test_score

#         # Filter trials by those that are above the test score threshold and have a train score not lower than the test score
#         filtered_trials = [t for t in trials_data if (t["mean_train_score"] >= test_score_threshold)]

#         # Select the trial with the smallest average gap score
#         if filtered_trials:
#             gso_1_trial = min(filtered_trials, key=lambda x: np.mean(x["gap_scores"]))
#             return gso_1_trial["params"]
#         else:
#             return trials[0].params

# def _one_sem_model(trials, model_name, samples, splits, method):
#     """
#     This function selects the 'simplest' hyperparameters for the given model.
#     It does this by selecting the hyperparameters that resulted validation score higher than the SEM threshold and have the smallest complexity score which is calculated differently in each estimator.
    
#     Parameters
#     ----------
#     trials : list of optuna.trial.Trial
#         List of trials to select from.
#     model_name : str
#         Name of the model to select hyperparameters for.
#     samples : int
#         Number of samples in the dataset.
#     splits : int
#         Number of splits in the cross-validation.
#     method : str
#         Method to use for selecting the simplest hyperparameters. Can be 'one_sem' or 'one_sem_grd'.
    
#     Returns
#     -------
#     dict
#         Selected hyperparameters.
#     """
#     constraints = hyper_compl[model_name]
    
#     trials_data = [
#         {
#             "params": t.params,
#             "value": t.values[0],
#             "sem": t.user_attrs["std_test_score"] / (splits**0.5),
#             "train_time": t.user_attrs["mean_fit_time"],
#         }
#         for t in trials
#         if t.state == optuna.trial.TrialState.COMPLETE
#     ]
#     trials_data = sorted(
#         trials_data, key=lambda x: (x["value"], -x["sem"]), reverse=True
#     )

#     # Find the best score and its SEM value
#     best_score = trials_data[0]["value"]
#     best_sem_score = trials_data[0]["sem"]

#     # Find the scores that will possibly return simpler models with equally good performance
#     sem_threshold = best_score - best_sem_score
#     filtered_trials = [t for t in trials_data if t["value"] >= sem_threshold]
#     if method == "one_sem":
#         # Calculate complexity for each filtered trial
#         for trial in filtered_trials:
#             trial["complexity"] = _calculate_complexity(trial, model_name, samples)

#         # Find the trial with the smallest complexity
#         shorted_trials = sorted(filtered_trials, key=lambda x: (x["complexity"], x["train_time"]))
#         best_trial = shorted_trials[0]

#         return best_trial["params"]
#     elif method == "one_sem_grd":
#         # Retrieve the hyperparameter priorities for the given model type
#         hyperparams = hyper_compl[model_name]

#         # Iterate over the hyperparameters and their sorting orders
#         for hyper, order in hyperparams.items():
#             # Sort the models based on the current hyperparameter
#             sorted_dict = sorted(filtered_trials, key=lambda x: x['params'][hyper], reverse=not order)

#             # Get the best value for the current hyperparameter from the sorted list
#             best_value = sorted_dict[0]['params'][hyper]

#             # Find all models with the best value for the current hyperparameter
#             models_with_same_hyper = []
#             for model in sorted_dict:
#                 if model['params'][hyper] == best_value:
#                     models_with_same_hyper.append(model)

#             # If there is only one model with the best hyperparameter value, return it
#             if len(models_with_same_hyper) == 1:
#                 filtered_trials = [models_with_same_hyper[0]].copy()
#                 break
#             else:
#                 # Otherwise, update all_models to only include models with the best hyperparameter value
#                 filtered_trials = models_with_same_hyper.copy()

#         # If multiple models have the same best hyperparameter values, return the first one
#         simple_model = filtered_trials[0]

#         return simple_model["params"]
    
    
# def _calculate_complexity(trial, model_name, samples):
#     """
#     This function calculates the complexity of a model based on its hyperparameters for each estimator.
    
#     Parameters
#     ----------
#     trial : dict
#         Trial to calculate complexity for.
#     model_name : str
#         Name of the model to calculate complexity for.
#     samples : int
#         Number of samples in the dataset.
    
#     Returns
#     -------
#     float
#         Calculated complexity.
#     """
#     params = trial["params"]
#     if model_name in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier', 'LGBMClassifier']:
#         max_depth = params["max_depth"]
#         if model_name == 'RandomForestClassifier' or model_name == 'GradientBoostingClassifier':
#             actual_depth = min((samples / params["min_samples_leaf"]), max_depth)
#         elif model_name == 'XGBClassifier':
#             actual_depth = max_depth  # Assuming XGBClassifier does not use min_samples_leaf
#         elif model_name == 'LGBMClassifier':
#             actual_depth = min((samples / params["min_child_samples"]), max_depth)
#         complexity = params["n_estimators"] * (2 ** (actual_depth - 1))
#     elif model_name == 'CatBoostClassifier':
#         max_depth = params["depth"]
#         actual_depth = min((samples / params["min_data_in_leaf"]), max_depth)
#         complexity = params["n_estimators"] * (2 ** (actual_depth - 1))#*params["iterations"]
#     elif model_name == 'LogisticRegression' or model_name == 'ElasticNet':
#         complexity = params["C"] * params["max_iter"]
#     elif model_name == 'SVC':
#         if params["kernel"] == 'poly':
#             complexity = params["C"] * params["degree"]
#         else:
#             complexity = params["C"]
#     elif model_name == 'KNeighborsClassifier':
#         complexity = params["leaf_size"]
#     elif model_name == 'LinearDiscriminantAnalysis':
#         complexity = params["shrinkage"]
#     elif model_name == 'GaussianNB':
#         complexity = params["var_smoothing"]
#     elif model_name == 'GaussianProcessClassifier':
#         complexity = params["max_iter_predict"]*params["n_restarts_optimizer"]
#     else:
#         complexity = float('inf')  # If model not recognized, set high complexity
#     return complexity