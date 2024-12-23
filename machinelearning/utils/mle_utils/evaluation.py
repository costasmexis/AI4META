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

from machinelearning.utils.calc_fnc import _calc_shap, _calculate_metrics
from machinelearning.utils.balance_fnc import _class_balance

def _evaluate(
    X, y, best_model, best_params, config
    ):
    """
    Evaluate the performance of a machine learning model using cross-validation or bootstrap methods.

    :param X: The input features.
    :type X: pandas.DataFrame
    :param y: The target variable.
    :type y: pandas.Series
    :param cv: The number of cross-validation folds.
    :type cv: int
    :param evaluation: The evaluation method to use. Must be either 'cv_simple', 'bootstrap', or 'cv_rounds'.
    :type evaluation: str
    :param rounds: The number of rounds for cross-validation or bootstrap evaluation.
    :type rounds: int
    :param best_model: The best model obtained from model selection.
    :type best_model: object
    :param best_params: The best hyperparameters obtained from model selection.
    :type best_params: dict
    :param way: The model selection method used. Only required if evaluation is 'cv_rounds' or 'cv_simple'.
    :type way: object
    :param calculate_shap: Whether to calculate SHAP values or not.
    :type calculate_shap: bool
    :raises ValueError: If cv is less than 2.
    :raises ValueError: If evaluation method is not one of 'cv_simple', 'bootstrap', or 'cv_rounds'.
    :return: The best model, evaluation results, and SHAP values (if calculate_shap is True).
    :rtype: tuple
    """
    
    # Initiate shap values array
    x_shap = np.zeros((X.shape[0], X.shape[1]))

    if config['evaluation'] == "cv_rounds":
        extra_metrics_scores, x_shap = _cv_rounds_validation(X, y, best_model, config, x_shap)
    
    elif config['evaluation'] == "bootstrap":
        extra_metrics_scores = _bootstrap_validation(X, y, best_model, config['extra_metrics'])#, calculate_shap=False)

    elif config['evaluation'] == "oob":
        extra_metrics_scores = _oob_validation(X, y, best_model, config['extra_metrics'])#, calculate_shap=False)
    
    elif config['evaluation'] == "train_test":
        extra_metrics_scores = _train_test_validation(X, y, best_model, config['class_balance'], config['extra_metrics'])#, calculate_shap=False)
                
    local_data_full_outer = pd.DataFrame(extra_metrics_scores)

    return local_data_full_outer, x_shap

def _cv_rounds_validation(X, y, model, config, x_shap):
    extra_metrics_scores = {extra: [] for extra in config['extra_metrics']} if config['extra_metrics'] else {}    
    for i in range(config['rounds']):
        cv_splits = StratifiedKFold(n_splits=config['splits'], shuffle=True, random_state=i)
        temp_train_test_indices = list(cv_splits.split(X, y))
        for train_index, test_index in temp_train_test_indices:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            # y_pred = model.predict(X_test)
            
            # Calculate and store scores for each extra metric
            # if config['extra_metrics'] is not None:
            #     for extra in config['extra_metrics']:
            #         extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
            #         extra_metrics_scores[extra].append(extra_score)
            extra_metrics_scores = _calculate_metrics(config['extra_metrics'], extra_metrics_scores, model, X_test, y_test)

            if config['calculate_shap']:
                shap_values = _calc_shap(X_train, X_test, model)
                x_shap[test_index, :] = np.add(
                    x_shap[test_index, :], shap_values.values
                )
    
    if config['calculate_shap']:
        x_shap = x_shap / (config['rounds'])

    return extra_metrics_scores, x_shap

def _bootstrap_validation(
        X, y, model, extra_metrics=None):#, calculate_shap=False
    """Performs bootstrap validation for model evaluation.
    :return: A tuple of (bootstrap_scores, extra_metrics_scores).
    :rtype: tuple
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True
    )

    bootstrap_scores = []
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}    
    
    for i in tqdm(range(100), desc="Bootstrap validation"):
        model_bootstrap = copy.deepcopy(model)
        X_train_res, y_train_res = resample(X_train, y_train, random_state=i)
        model_bootstrap.fit(X_train_res, y_train_res)
        # y_pred = model_bootstrap.predict(X_test)

        # Calculate and store extra metrics
        extra_metrics_scores = _calculate_metrics(extra_metrics, extra_metrics_scores, model_bootstrap, X_test, y_test)

    return extra_metrics_scores#, bootstrap_scores, 

def _oob_validation(
        X, y, model, extra_metrics=None
    ):

    oob_scores = []
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    
    for i in tqdm(range(100), desc="OOB validation"):
        # Generate a random bootstrap sample with replacement
        rng = np.random.default_rng(i)  # New random generator for each seed
        indices = rng.choice(range(X.shape[0]), size=X.shape[0], replace=True)

        X_train, y_train = X.iloc[indices, :], y[indices]
        
        # Determine the OOB indices
        oob_indices = list(set(range(X.shape[0])) - set(indices))
        X_test, y_test = X.iloc[oob_indices, :], y[oob_indices]
        
        model_oob = copy.deepcopy(model)
        model_oob = model_oob.fit(X_train, y_train)
        y_pred = model_oob.predict(X_test)
        
        # Calculate and store extra metrics
        # if extra_metrics is not None:
        #     for extra in extra_metrics:
        #         extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
        #         extra_metrics_scores[extra].append(extra_score)
        extra_metrics_scores = _calculate_metrics(extra_metrics, extra_metrics_scores, model_oob, X_test, y_test)

    return  extra_metrics_scores#, oob_scores,

def _train_test_validation(X, y, model, class_balance_method, extra_metrics=None):
    tt_prop_scores = []
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

    for i, (train_index, test_index) in tqdm(enumerate(sss.split(X, y)), desc="TT prop validation"):
        # Use .iloc for DataFrame X and NumPy indexing for array y
        X_train, y_train = _class_balance(X.iloc[train_index], y[train_index], class_balance_method), _class_balance(X.iloc[test_index], y[test_index], class_balance_method)
        X_test, y_test = X.iloc[test_index], y[test_index]

        # Deepcopy the model and fit it on the train set
        model_tt_prop = model.fit(X_train, y_train)  
        y_pred = model_tt_prop.predict(X_test)

        # Calculate and store extra metrics
        # if extra_metrics is not None:
        #     for extra in extra_metrics:
        #         extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
        #         extra_metrics_scores[extra].append(extra_score)
        extra_metrics_scores = _calculate_metrics(extra_metrics, extra_metrics_scores, model_tt_prop, X_test, y_test)

    return extra_metrics_scores, #tt_prop_scores, 