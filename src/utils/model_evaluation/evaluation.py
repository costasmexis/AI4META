import numpy as np
import pandas as pd
from sklearn.utils import resample
import sklearn.metrics as metrics
from tqdm import tqdm
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    StratifiedShuffleSplit
)
import copy

from src.utils.metrics.metrics import _calculate_metrics
from src.utils.metrics.shap import _calc_shap
from src.data_manipulation.class_balance import _class_balance

def _evaluate(X, y, best_model, best_params, config):
    """
    Evaluate the performance of a machine learning model using various validation methods.

    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target labels.
    best_model : object
        Trained machine learning model.
    best_params : dict
        Best hyperparameters from model selection.
    config : dict
        Configuration dictionary containing evaluation settings.

    Returns:
    --------
    tuple
        local_data_full_outer : pandas.DataFrame
            DataFrame containing evaluation metrics.
        x_shap : numpy.ndarray
            SHAP values array if SHAP is calculated, otherwise zeros.
    """
    x_shap = np.zeros((X.shape[0], X.shape[1]))

    if config['evaluation'] == "cv_rounds":
        extra_metrics_scores, x_shap = _cv_rounds_validation(X, y, best_model, config, x_shap)
    elif config['evaluation'] == "bootstrap":
        extra_metrics_scores = _bootstrap_validation(X, y, best_model, config['extra_metrics'])
    elif config['evaluation'] == "oob":
        extra_metrics_scores = _oob_validation(X, y, best_model, config['extra_metrics'])
    elif config['evaluation'] == "train_test":
        extra_metrics_scores = _train_test_validation(X, y, best_model, config['class_balance'], config['extra_metrics'])
    else:
        raise ValueError("Invalid evaluation method specified.")

    local_data_full_outer = pd.DataFrame(extra_metrics_scores)
    return local_data_full_outer, x_shap

def _cv_rounds_validation(X, y, model, config, x_shap):
    """
    Perform cross-validation rounds for model evaluation.

    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target labels.
    model : object
        Machine learning model.
    config : dict
        Configuration dictionary.
    x_shap : numpy.ndarray
        Array to store SHAP values.

    Returns:
    --------
    tuple
        extra_metrics_scores : dict
            Scores for extra metrics.
        x_shap : numpy.ndarray
            Updated SHAP values.
    """
    extra_metrics_scores = {extra: [] for extra in config['extra_metrics']} if config['extra_metrics'] else {}
    
    for i in range(config['rounds']):
        cv_splits = StratifiedKFold(n_splits=config['splits'], shuffle=True, random_state=i)
        temp_train_test_indices = list(cv_splits.split(X, y))
        
        for train_index, test_index in temp_train_test_indices:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)

            extra_metrics_scores = _calculate_metrics(config['extra_metrics'], extra_metrics_scores, model, X_test, y_test)

            if config['calculate_shap']:
                shap_values = _calc_shap(X_train, X_test, model)
                x_shap[test_index, :] += shap_values.values

    if config['calculate_shap']:
        x_shap /= config['rounds']

    return extra_metrics_scores, x_shap

def _bootstrap_validation(X, y, model, extra_metrics=None):
    """
    Perform bootstrap validation for model evaluation.

    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target labels.
    model : object
        Machine learning model.
    extra_metrics : list, optional
        List of additional metrics to calculate.

    Returns:
    --------
    dict
        Scores for extra metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    
    for i in tqdm(range(100), desc="Bootstrap validation"):
        model_bootstrap = copy.deepcopy(model)
        X_train_res, y_train_res = resample(X_train, y_train, random_state=i)
        model_bootstrap.fit(X_train_res, y_train_res)

        extra_metrics_scores = _calculate_metrics(extra_metrics, extra_metrics_scores, model_bootstrap, X_test, y_test)

    return extra_metrics_scores

def _oob_validation(X, y, model, extra_metrics=None):
    """
    Perform out-of-bag (OOB) validation for model evaluation.

    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target labels.
    model : object
        Machine learning model.
    extra_metrics : list, optional
        List of additional metrics to calculate.

    Returns:
    --------
    dict
        Scores for extra metrics.
    """
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}

    for i in tqdm(range(100), desc="OOB validation"):
        rng = np.random.default_rng(i)
        indices = rng.choice(range(X.shape[0]), size=X.shape[0], replace=True)

        X_train, y_train = X.iloc[indices, :], y[indices]
        oob_indices = list(set(range(X.shape[0])) - set(indices))
        X_test, y_test = X.iloc[oob_indices, :], y[oob_indices]

        model_oob = copy.deepcopy(model)
        model_oob.fit(X_train, y_train)
        extra_metrics_scores = _calculate_metrics(extra_metrics, extra_metrics_scores, model_oob, X_test, y_test)

    return extra_metrics_scores

def _train_test_validation(X, y, model, class_balance_method, extra_metrics=None):
    """
    Perform train-test split validation for model evaluation.

    Parameters:
    -----------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target labels.
    model : object
        Machine learning model.
    class_balance_method : str
        Method for class balancing.
    extra_metrics : list, optional
        List of additional metrics to calculate.

    Returns:
    --------
    dict
        Scores for extra metrics.
    """
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

    for i, (train_index, test_index) in tqdm(enumerate(sss.split(X, y)), desc="TT prop validation"):
        X_train, y_train = _class_balance(X.iloc[train_index], y[train_index], class_balance_method)
        X_test, y_test = X.iloc[test_index], y[test_index]

        model_tt_prop = model.fit(X_train, y_train)
        extra_metrics_scores = _calculate_metrics(extra_metrics, extra_metrics_scores, model_tt_prop, X_test, y_test)

    return extra_metrics_scores
