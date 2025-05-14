import logging
from optuna.integration import OptunaSearchCV
from optuna.logging import set_verbosity
import copy
import numpy as np
import pandas as pd
from typing import Union

from src.utils.metrics.metrics import _calculate_metrics
from src.utils.model_manipulation.inner_selection import _one_sem_model, _gso_model
from src.utils.model_manipulation.model_instances import _create_model_instance
from src.features.sfm import _sfm
from src.constants.parameters_grid import optuna_grid
from src.constants.translators import AVAILABLE_CLFS, SFM_COMPATIBLE_ESTIMATORS
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

            # Normalize the data
            X_train_norm, y_train_norm, X_test_norm, feature_indicator = processor.process_data(
                X_train, y_train, X_test, num_feature2_use, features_name_list=None, random_state=i
            )

            for estimator_name in config.clfs:
                estimator = AVAILABLE_CLFS[estimator_name]

                # Handle SelectFromModel (SFM) for supported classifiers
                if config.sfm and estimator_name in SFM_COMPATIBLE_ESTIMATORS and feature_indicator!= "none":
                    X_train_norm_sfm, y_train_sfm, X_test_norm_sfm, _ = processor.process_data(
                        X_train, y_train, X_test, num_feature2_use, features_name_list=None, random_state=i, sfm=config.sfm, estimator_name=estimator_name
                    )

                    Xtrft, Xteft, ytr = X_train_norm_sfm, X_test_norm_sfm, y_train_sfm
                else: 
                    Xtrft, Xteft, ytr = X_train_norm, X_test_norm, y_train_norm
                
                if method == 'rcv_accel':
                    _fit_cvdefault(
                        Xtrft,
                        Xteft,
                        ytr,
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
                        ytr,
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

def _store_results(
        results: dict, 
        num_feature: Union[int, str],
        estimator_name: str, 
        sfm: bool, 
        feature_selection_type: str, 
        X_train_selected: pd.DataFrame, 
        method: str
    ):
    """
    Stores the results of a model selection process into a dictionary.

    Args:
        results (dict): A dictionary to store the results. Keys include:
            - "Selected_Features": List of selected feature names or None.
            - "Number_of_Features": Number of selected features.
            - "Way_of_Selection": Method used for feature selection.
            - "Classifiers": Name of the classifier with feature selection details.
            - "MS_Method": Method used for model selection.
        num_feature (str or int): Number of features selected. Use "none" if no feature selection is applied.
        estimator_name (str): Name of the classifier or estimator.
        sfm (bool): Indicates if SelectFromModel (sfm) was used for feature selection.
        feature_selection_type (str): Type of feature selection method used (e.g., "PCA", "RFE").
        X_train_selected (pd.DataFrame): The training data after feature selection.
        method (str): The method used for model selection.

    Returns:
        None: The function modifies the `results` dictionary in place.
    """
    results["Selected_Features"].append(None if num_feature == "none" else X_train_selected.columns.tolist())
    results["Number_of_Features"].append(X_train_selected.shape[1] if num_feature == "none" else num_feature)
    results["Way_of_Selection"].append("none" if num_feature == "none" else ("sfm" if sfm and estimator_name in SFM_COMPATIBLE_ESTIMATORS else feature_selection_type))
    results["Classifiers"].append(estimator_name if num_feature == "none" else f"{estimator_name}_{'sfm' if sfm and estimator_name in SFM_COMPATIBLE_ESTIMATORS else feature_selection_type}_{num_feature}")
    results["MS_Method"].append(method)


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
    num_feature: Union[int, str],
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
        cv=config.inner_splits,
        return_train_score=True,
        n_jobs=n_jobs,
        verbose=0,
        n_trials=config.n_trials
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
            results, num_feature, estimator_name, config.sfm, config.feature_selection_type, X_train, 'NestedCV'
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
    num_feature: Union[int, str],
    test_index: np.ndarray,
    n_samples: int
):
    # Train the model without hyperparameter optimization
    res_model = _create_model_instance(estimator_name, params=None)
    results["Estimator"].append(estimator_name)
    res_model.fit(X_train, y_train)
    results = _calculate_metrics(config.extra_metrics, results, res_model, X_test, y_test)
    _store_results(
        results, num_feature, estimator_name, config.sfm, config.feature_selection_type, X_train, 'DefaultCV'
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