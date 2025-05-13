import logging
from optuna.integration import OptunaSearchCV
from optuna.logging import set_verbosity
import copy
import numpy as np
import pandas as pd

from src.utils.metrics.metrics import _calculate_metrics
from src.utils.model_manipulation.inner_selection import _one_sem_model, _gso_model
from src.utils.model_manipulation.model_instances import _create_model_instance
from src.features.sfm import _sfm, _sfm_condition
from src.constants.parameters_grid import optuna_grid
from src.constants.translators import AVAILABLE_CLFS
from src.data.process import DataProcessor
from src.utils.validation.dataclasses import ModelSelectionConfig as Config

def _fit(
        X: pd.DataFrame, 
        y: np.ndarray, 
        config: Config, 
        results: dict, 
        train_index: np.ndarray,
        test_index: np.ndarray, 
        fit_type: str,
        i: int,
        n_jobs: int = 1,
        processor: DataProcessor = None
    ):
    """
    Perform the fitting procedure for the machine learning pipeline.

    This function loops over the number of features and classifiers, applies preprocessing,
    handles feature selection, and trains models with or without hyperparameter optimization.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature dataset.
    y : pandas.Series or array-like
        Target labels.
    config : dict
        Configuration dictionary containing hyperparameters and other settings.
    results : dict
        Dictionary to store the results of the pipeline.
    train_index : array-like
        Indices for the training set.
    test_index : array-like
        Indices for the testing set.
    i : int
        Current round number for reproducibility.
    n_jobs : int or 1, optional
        Number of jobs for parallel execution. Defaults to None.
    """
    if fit_type == "both":
        fit_method = ['rcv_accel','rnested_cv']
    else: 
        fit_method = [fit_type]

    for method in fit_method:
        for num_feature2_use in config.num_features:
            # Split the data into training and testing sets
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            all_features = num_feature2_use!=X.shape[1]

            # # Preprocess data (feature selection, normalization, etc.)
            # if (config['sfm']) and (all_features):
            #     # If SFM is used, we need to select features first
            #     X_train_norm, X_test_norm, num_feature = preprocess.(
            #         config, None, X_train, y_train, X_test
            #     )
            #     # Apply class balancing
            #     X_train_norm, y_train = _class_balance(X_train_norm, y_train, config["class_balance"], i)

            X_train_norm, y_train_norm, X_test_norm, feature_indicator = processor.process_data(
                X_train, y_train, X_test, num_feature2_use, features_names_list=None, random_state=i
            )

            for estimator_name in config.clfs:
                estimator = AVAILABLE_CLFS[estimator_name]

                # # Handle SelectFromModel (SFM) for supported classifiers
                # if _sfm_condition(config['sfm'], estimator_name, all_features):
                #     # If SFM is used, we need to select features first
                #     X_train_selected_sfm, X_test_selected_sfm, num_feature = _sfm(
                #         estimator, X_train_norm, X_test_norm, y_train, num_feature2_use
                #     )
                Xtrft, Xteft = X_train_norm, X_test_norm
                
                if method == 'rcv_accel':
                    _fit_cvdefault(
                        Xtrft,
                        Xteft,
                        y_train,
                        y_test,
                        config,
                        results,
                        estimator_name,
                        estimator,
                        all_features,
                        feature_indicator,
                        test_index,
                        y.shape[0]
                    )
                else:
                    _fit_nested(
                        Xtrft,
                        Xteft,
                        y_train,
                        y_test,
                        config,
                        results,
                        i,
                        n_jobs,
                        estimator_name,
                        estimator,
                        all_features,
                        feature_indicator,
                        test_index,
                        y.shape[0]
                    )
                    
        return results

def _store_results(results, num_feature, estimator_name, sfm, feature_selection_type, X_train_selected):
    """
    Helper function to store results for each model and feature selection.

    Parameters:
    -----------
    results : dict
        Dictionary to store the results.
    num_feature : int or str
        Number of features selected.
    estimator_name : str
        Name of the estimator used.
    config : dict
        Configuration dictionary.
    X_train_selected : pandas.DataFrame
        The training dataset after feature selection.
    """
    if num_feature == "none" or num_feature is None:
        results["Selected_Features"].append(None)
        results["Number_of_Features"].append(X_train_selected.shape[1])
        results["Way_of_Selection"].append("none")
        results["Classifiers"].append(estimator_name)
    else:
        fs_type = "sfm" if sfm else feature_selection_type
        results["Classifiers"].append(f"{estimator_name}_{fs_type}_{num_feature}")
        results["Selected_Features"].append(X_train_selected.columns.tolist())
        results["Number_of_Features"].append(num_feature)
        results["Way_of_Selection"].append(fs_type)

    # # Debugging: Log the current results state
    # print(f"Updated results for {estimator_name}: {results}")

def _fit_nested(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        config: Config,
        results: dict,
        i: int,
        n_jobs: int,
        estimator_name: str,
        estimator: object,
        all_features: bool,
        num_feature: int,
        test_index: np.ndarray,
        n_samples: int
):
    # Logging setup
    set_verbosity(logging.ERROR)
    logging.getLogger("optuna").setLevel(logging.ERROR)

    # Hyperparameter optimization with Optuna
    opt_grid = "NestedCV"

    clf = OptunaSearchCV(
        estimator=estimator,
        scoring=config.inner_scoring,
        param_distributions=optuna_grid[opt_grid][estimator_name],
        cv=config.inner_cv,
        return_train_score=True,
        n_jobs=n_jobs,
        verbose=0,
        n_trials=config.n_trials,
    )
    
    clf.fit(X_train, y_train)
    trials = clf.study_.trials

    for inner_selection in config.inner_selection:
        results['Inner_selection_mthd'].append(inner_selection)
        results['Estimator'].append(estimator_name)

        if inner_selection == "validation_score":
            res_model = copy.deepcopy(clf)
            params = res_model.study_.best_params
        else:
            if inner_selection in ["one_sem", "one_sem_grd"]:
                params = _one_sem_model(trials, estimator_name, len(X_train), config.inner_splits, inner_selection)
            elif inner_selection in ["gso_1", "gso_2"]:
                params = _gso_model(trials, estimator_name, config.inner_splits, inner_selection)

        res_model = _create_model_instance(estimator_name, params)
        res_model.fit(X_train, y_train)

        results["Hyperparameters"].append(params)

        # Evaluate metrics depending on SFM condition
        results = _calculate_metrics(config.extra_metrics, results, res_model, X_test, y_test)
        # Store feature selection and results
        _store_results(
            results, num_feature, estimator_name, config, X_train
        )
        # Track predictions for Samples_counts
        y_pred = res_model.predict(X_test)

        samples_counts = np.zeros(n_samples)
        for idx, resu, pred in zip(test_index, y_test, y_pred):
            if pred == resu:
                samples_counts[idx] += 1
        results["Samples_counts"].append(samples_counts.tolist())

        # Debugging: Check consistency of results
        lengths = {key: len(val) for key, val in results.items()}
        if len(set(lengths.values())) > 1:
            print("Inconsistent lengths in results:", lengths)
            raise ValueError("Inconsistent lengths in results dictionary")
          
def _fit_cvdefault(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        config: Config,
        results: dict,
        estimator_name: str,
        estimator: object,
        all_features: bool,
        num_feature: int,
        test_index: np.ndarray,
        n_samples: int
    ):
        # Train the model without hyperparameter optimization
        res_model = _create_model_instance(estimator_name, params=None)
        results["Estimator"].append(estimator_name)
        res_model.fit(X_train, y_train)
        results = _calculate_metrics(config.extra_metrics, results, res_model, X_test, y_test)
        _store_results(
            results, num_feature, estimator_name, config.sfm, config.feature_selection_type, X_train
        )
        # Track predictions for Samples_counts
        y_pred = res_model.predict(X_test)

        results["Hyperparameters"].append('Default')
        results['Inner_selection_mthd'].append('validation_score')

        # Track predictions for Samples_counts
        samples_counts = np.zeros(n_samples)
        for idx, resu, pred in zip(test_index, y_test, y_pred):
            if pred == resu:
                samples_counts[idx] += 1
        results["Samples_counts"].append(samples_counts.tolist())

        # Debugging: Check consistency of results
        lengths = {key: len(val) for key, val in results.items()}
        if len(set(lengths.values())) > 1:
            print("Inconsistent lengths in results:", lengths)
            raise ValueError("Inconsistent lengths in results dictionary")