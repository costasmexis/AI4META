import numpy as np
import optuna
from src.constants.parameters_grid import hyper_compl

def _gso_model(
        trials: list, 
        model_name: str, 
        splits: int, 
        method: str
    ) -> dict:
    """
    Select optimal hyperparameters using Gap Score Optimization (GSO) approach.

    This method identifies hyperparameters that achieve a balance between model performance
    and generalization by analyzing the gap between training and test scores.

    Parameters
    ----------
    trials : list[optuna.Trial]
        List of completed Optuna trials containing model performance data.
    model_name : str
        Name of the model being optimized.
    splits : int
        Number of cross-validation splits used.
    method : str
        GSO method to use, either 'gso_1' (train score based) or 'gso_2' (test score based).

    """
    # Validate method parameter
    if method not in ["gso_1", "gso_2"]:
        raise ValueError(f"Invalid method: {method}. Must be either 'gso_1' or 'gso_2'")

    # Extract and process trial data
    trials_data = [
        {
            "params": t.params,
            "mean_test_score": t.user_attrs.get("mean_test_score"),
            "mean_train_score": t.user_attrs.get("mean_train_score"),
            "gap_scores": [
                abs(t.user_attrs.get(f"split{i}_test_score") - 
                    t.user_attrs.get(f"split{i}_train_score"))
                for i in range(splits)
            ],
        }
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # Select reference score based on method
    key_score = "mean_train_score" if method == "gso_1" else "mean_test_score"
    
    # Sort trials by the chosen score metric
    trials_data.sort(key=lambda x: x[key_score], reverse=True)
    best_score = trials_data[0][key_score]
    
    # Filter trials above the threshold (85% of best score)
    score_threshold = 0.85 * best_score
    filtered_trials = [t for t in trials_data if t[key_score] >= score_threshold]

    # Select hyperparameters with minimal gap score
    if filtered_trials:
        best_trial = min(filtered_trials, key=lambda x: np.mean(x["gap_scores"]))
        return best_trial["params"]

    # Fallback to best overall parameters if no trials meet criteria
    return trials[0].params

def _calculate_complexity(
        trial: dict, 
        model_name: str, 
        samples: int
    ) -> float:
    """
    Calculate the computational complexity score for a model configuration.

    Different model types have different complexity metrics based on their
    hyperparameters and characteristics.

    Parameters
    ----------
    trial : dict
        Dictionary containing model parameters and trial information.
    model_name : str
        Name of the model being evaluated.
    samples : int
        Number of samples in the dataset.
    """
    params = trial["params"]

    # Complexity calculation for tree-based models
    if model_name in ["RandomForestClassifier", "XGBClassifier", 
                     "GradientBoostingClassifier", "LGBMClassifier"]:
        max_depth = params.get("max_depth", float("inf"))
        leaf_estimator = params.get("min_samples_leaf", 
                                  params.get("min_child_samples", 1))
        actual_depth = min(samples / leaf_estimator, max_depth)
        return params["n_estimators"] * (2 ** (actual_depth - 1))

    # Complexity for CatBoost
    elif model_name == "CatBoostClassifier":
        max_depth = params.get("depth", float("inf"))
        actual_depth = min(samples / params.get("min_data_in_leaf", 1), max_depth)
        return params["n_estimators"] * (2 ** (actual_depth - 1))

    # Complexity for linear models
    elif model_name in ["LogisticRegression", "ElasticNet"]:
        return params["C"] * params["max_iter"]

    # Complexity for SVM
    elif model_name == "SVC":
        return (params["C"] * params["degree"] 
                if params.get("kernel") == "poly" 
                else params["C"])

    # Complexity for other models
    elif model_name == "KNeighborsClassifier":
        return params["leaf_size"]
    elif model_name == "LinearDiscriminantAnalysis":
        return params.get("shrinkage", 1.0)
    elif model_name == "GaussianNB":
        return params.get("var_smoothing", 1.0)
    elif model_name == "GaussianProcessClassifier":
        return (params.get("max_iter_predict", 1) * 
                params.get("n_restarts_optimizer", 1))

    # Default complexity for unsupported models
    return float("inf")

def _one_sem_model(
       trials: list,
       model_name: str,
       samples: int,
       splits: int,
       method: str) -> dict:
    """
    Select model hyperparameters using the one-standard-error rule with variations.

    This method implements both traditional one-standard-error rule and a gradient-based
    version for hyperparameter selection, aiming to choose the simplest model whose
    performance is within one standard error of the best model.

    Parameters
    ----------
    trials : list[optuna.Trial]
        List of completed Optuna trials.
    model_name : str
        Name of the model being optimized.
    samples : int
        Number of samples in the dataset.
    splits : int
        Number of cross-validation splits.
    method : str
        Selection method: 'one_sem' (traditional) or 'one_sem_grd' (gradient-based).
    """
    # Validate method parameter
    if method not in ["one_sem", "one_sem_grd"]:
        raise ValueError(
            f"Invalid method: {method}. Must be either 'one_sem' or 'one_sem_grd'"
        )

    # Process trial data
    trials_data = [
        {
            "params": t.params,
            "value": t.values[0],
            "sem": t.user_attrs["std_test_score"] / (splits**0.5),
        }
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # Sort trials by performance and SEM
    trials_data.sort(key=lambda x: (x["value"], -x["sem"]), reverse=True)

    # Calculate threshold based on one standard error rule
    best_score = trials_data[0]["value"]
    sem_threshold = best_score - trials_data[0]["sem"]
    
    # Filter trials above the threshold
    filtered_trials = [t for t in trials_data if t["value"] >= sem_threshold]

    if method == "one_sem":
        # Calculate complexity for each trial and select simplest
        for trial in filtered_trials:
            trial["complexity"] = _calculate_complexity(trial, model_name, samples)
        simplest_trial = min(filtered_trials, key=lambda x: x["complexity"])
        return simplest_trial["params"]
    else:
        # Apply gradient-based selection using predefined hyperparameter priorities
        hyperparams = hyper_compl[model_name]
        for hyper, order in hyperparams.items():
            filtered_trials.sort(key=lambda x: x["params"][hyper], reverse=not order)
        return filtered_trials[0]["params"]
