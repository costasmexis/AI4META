import time
import logging
import optuna
import copy
import numpy as np

from machinelearning.utils.calc_fnc import _calculate_metrics
from machinelearning.utils.inner_selection_fnc import _one_sem_model, _gso_model
from machinelearning.utils.balance_fnc import _class_balance
from machinelearning.utils.modinst_fnc import _create_model_instance
from machinelearning.utils.features_selection import _preprocess, _sfm
from machinelearning.utils.optuna_grid import optuna_grid
from machinelearning.utils.translators import AVAILABLE_CLFS

def _set_optuna_verbosity(level):
    """ Adjust Optuna's verbosity level """
    optuna.logging.set_verbosity(level)
    logging.getLogger("optuna").setLevel(level)

def _fit_procedure(X, y, config, results, train_index, test_index, i, n_jobs=None):
    """
    This function is used to perform the fit procedure of the machine learning pipeline. 
    It loops over the number of features and the classifiers given in the config dictionary.

    Parameters
    ----------
    config : dict
        A dictionary containing all the configuration for the machine learning pipeline.
    results : dict
        A dictionary containing the results of the machine learning pipeline.
    train_index : array-like
        The indices of the training set.
    test_index : array-like
        The indices of the testing set.
    i : int
        The current round of the cross-validation.
    n_jobs : int or None
        The number of jobs to run in parallel. If None, the number of jobs is set to the number of available CPU cores.

    Returns
    -------
    results : dict
        The results of the machine learning pipeline.
    """
    # Loop over the number of features
    for num_feature2_use in config["num_features"]:            
        X_train_selected, X_test_selected, num_feature = _preprocess(
            X, y, num_feature2_use, config, train_index=train_index, test_index=test_index 
        )
        y_train, y_test = y[train_index], y[test_index]
        
        # Apply class balancing strategies
        X_train_selected, y_train = _class_balance(X_train_selected, y_train, config["class_balance"], i)
                            
        # Check of the classifiers given list
        if config["clfs"] is None:
            raise ValueError("No classifier specified.")
        else:
            for estimator in config["clfs"]:
                # For every estimator find the best hyperparameteres
                name = estimator
                estimator = AVAILABLE_CLFS[estimator]
                if (config["sfm"]) and ((estimator == "RandomForestClassifier") or 
                            (estimator == "XGBClassifier") or 
                            (estimator == 'GradientBoostingClassifier') or 
                            (estimator == "LGBMClassifier") or 
                            (estimator == "CatBoostClassifier")) and (num_feature2_use != X.shape[1]):
                    
                    X_train_selected, X_test_selected, num_feature = _preprocess(
                        X, y, X.shape[1], config, train_index=train_index, test_index=test_index 
                    )
                    y_train, y_test = y[train_index], y[test_index]
                    # Check of the classifiers given list
                    # Perform feature selection using Select From Model (sfm)
                    X_train_selected, X_test_selected, num_feature = _sfm(estimator, X_train_selected, X_test_selected, y_train, num_feature2_use)

                    # Apply class balancing strategies
                    X_train_selected, y_train = _class_balance(X_train_selected, y_train, config["class_balance"], i)
                        
                if config['model_selection_type'] == 'rncv':
                    opt_grid = "NestedCV"
                    _set_optuna_verbosity(logging.ERROR)
                    clf = optuna.integration.OptunaSearchCV(
                        estimator=estimator,
                        scoring=config["inner_scoring"],
                        param_distributions=optuna_grid[opt_grid][name],
                        cv=config["inner_cv"],
                        return_train_score=True,
                        n_jobs=n_jobs,
                        verbose=0,
                        n_trials=config["n_trials"],
                    )
                    
                    clf.fit(X_train_selected, y_train)
                    
                    for inner_selection in config["inner_selection"]:
                        results['Inner_selection_mthd'].append(inner_selection)
                        # Store the results and apply one_sem method if its selected
                        results["Estimator"].append(name)
                        if inner_selection == "validation_score":
                            
                            res_model = copy.deepcopy(clf)

                            params = res_model.best_params_

                            trials = clf.trials_
                        else:
                            if (inner_selection == "one_sem") or (inner_selection == "one_sem_grd"):
                                samples = X_train_selected.shape[0]
                                # Find simpler parameters with the one_sem method if there are any
                                simple_model_params = _one_sem_model(trials, name, samples, config['inner_splits'],inner_selection)
                            elif (inner_selection == "gso_1") or (inner_selection == "gso_2"):
                                # Find parameters with the smaller gap score with gso_1 method if there are any
                                simple_model_params = _gso_model(trials, name, config['inner_splits'],inner_selection)

                            params = simple_model_params

                            # Fit the new model
                            new_params_clf = _create_model_instance(
                                name, simple_model_params
                            )
                            new_params_clf.fit(X_train_selected, y_train)

                            res_model = copy.deepcopy(new_params_clf)
                            
                        results["Hyperparameters"].append(params)
                        
                        # Metrics calculations
                        results = _calculate_metrics(config, results, res_model, X_test_selected, y_test)
                        
                        y_pred = res_model.predict(X_test_selected)

                        # Store the results using different names if feature selection is applied
                        if num_feature == "none" or num_feature is None:
                            results["Selected_Features"].append(None)
                            results["Number_of_Features"].append(X_test_selected.shape[1])
                            results["Way_of_Selection"].append("none")
                            results["Classifiers"].append(f"{name}")
                        else:
                            if (config["sfm"]) and ((estimator == "RandomForestClassifier") or 
                                (estimator == "XGBClassifier") or 
                                (estimator == 'GradientBoostingClassifier') or 
                                (estimator == "LGBMClassifier") or 
                                (estimator == "CatBoostClassifier")):
                                fs_type = "sfm"
                            else:
                                fs_type = config["feature_selection_type"]
                            results["Classifiers"].append(
                                f"{name}_{fs_type}_{num_feature}"
                            )
                            results["Selected_Features"].append(
                                X_train_selected.columns.tolist()
                            )
                            results["Number_of_Features"].append(num_feature)
                            results["Way_of_Selection"].append(
                                fs_type
                            )

                        # Track predictions
                        samples_counts = np.zeros(len(y))
                        for idx, resu, pred in zip(test_index, y_test, y_pred):
                            if pred == resu:
                                samples_counts[idx] += 1

                        results['Samples_counts'].append(samples_counts)
                        time.sleep(0.5)
                        
                else:
                    # Train the model
                    res_model = _create_model_instance(
                            name, params=None
                        )
                    
                    res_model.fit(X_train_selected, y_train)
                    results["Estimator"].append(name)

                    # Metrics calculations
                    results = _calculate_metrics(config, results, res_model, X_test_selected, y_test)
                    
                    y_pred = res_model.predict(X_test_selected)

                    # Store the results using different names if feature selection is applied
                    if num_feature == "none" or num_feature is None:
                        results["Selected_Features"].append(None)
                        results["Number_of_Features"].append(X_test_selected.shape[1])
                        results["Way_of_Selection"].append("none")
                        results["Classifiers"].append(f"{name}")
                    else:
                        if (config["sfm"]) and ((estimator == "RandomForestClassifier") or 
                            (estimator == "XGBClassifier") or 
                            (estimator == 'GradientBoostingClassifier') or 
                            (estimator == "LGBMClassifier") or 
                            (estimator == "CatBoostClassifier")):
                            fs_type = "sfm"
                        else:
                            fs_type = config["feature_selection_type"]
                        results["Classifiers"].append(
                            f"{name}_{fs_type}_{num_feature}"
                        )
                        results["Selected_Features"].append(
                            X_train_selected.columns.tolist()
                        )
                        results["Number_of_Features"].append(num_feature)
                        results["Way_of_Selection"].append(
                            fs_type
                        )

                    # Track predictions
                    samples_counts = np.zeros(len(y))
                    for idx, resu, pred in zip(test_index, y_test, y_pred):
                        if pred == resu:
                            samples_counts[idx] += 1

                    results['Samples_counts'].append(samples_counts)
                    time.sleep(0.5)
    return results