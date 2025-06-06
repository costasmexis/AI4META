from typing import Any, Dict, Tuple, Optional, Union
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
import logging
from src.utils.metrics.metrics import _calculate_metrics
from src.utils.metrics.shap import _calc_shap
from src.utils.validation.dataclasses import ModelEvaluationConfig as Config
from src.data.process import DataProcessor
import tqdm


# Global logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def _evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    best_model: Any,
    config: Config,
    processor: DataProcessor
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Evaluate a machine learning model using various evaluation methods.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : np.ndarray
        Target labels
    best_model : Any
        Trained machine learning model
    config : Dict
        Configuration dictionary containing evaluation settings

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        DataFrame containing evaluation metrics and SHAP values array

    Notes
    -----
    Supports multiple evaluation methods:
    - cv_rounds: Cross-validation with multiple rounds
    - bootstrap: Bootstrap evaluation
    - oob: Out-of-bag evaluation
    - train_test: Train-test split evaluation
    """
    x_shap = np.zeros((X.shape[0], X.shape[1]))

    if config.evaluation == "prefitted":
        extra_metrics_scores, x_shap = _prefitted_single_evaluation(X, y, best_model, config, x_shap)
    elif config.evaluation == "cv_rounds":
        extra_metrics_scores, x_shap = _cv_rounds_evaluation(X, y, best_model, config, x_shap)
    elif config.evaluation == "bootstrap":
        extra_metrics_scores = _bootstrap_evaluation(X, y, best_model, config.extra_metrics)
    elif config.evaluation == "oob":
        extra_metrics_scores = _oob_evaluation(X, y, best_model, config.extra_metrics)
    elif config.evaluation == "train_test":
        extra_metrics_scores = _train_test_evaluation(X, y, best_model, processor, config.extra_metrics)
    else:
        raise ValueError(f"Invalid evaluation method: {config.evaluation}")

    logger.info("✓ Evaluation completed")
    return pd.DataFrame(extra_metrics_scores), x_shap

def _prefitted_single_evaluation(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    config: Config,
    x_shap: Optional[np.ndarray]
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate a pre-fitted model on the provided dataset.

    This method calculates specified extra metrics and SHAP values
    for a model that has already been trained.
    """

    extra_metrics_scores = {extra: [] for extra in config.extra_metrics} if config.extra_metrics else {}

    extra_metrics_scores = _calculate_metrics(
        config.extra_metrics,
        extra_metrics_scores,
        model,
        X,
        y
    )
    logger.info("✓ Prefitted model metrics calculated")

    # If SHAP values are requested, calculate them
    if config.calculate_shap:
        shap_values = _calc_shap(X, X, model)
        x_shap += shap_values

    logger.info("✓ Prefitted evaluation complete")
    return extra_metrics_scores, x_shap

def _cv_rounds_evaluation(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    config: Config,
    x_shap: np.ndarray
) -> Tuple[Dict, np.ndarray]:
    """
    Perform cross-validation rounds for model evaluation.

    This method implements k-fold cross-validation repeated multiple times
    to get robust performance estimates.
    """    
    # Initialize metrics dictionary
    extra_metrics_scores = {extra: [] for extra in config.extra_metrics}
    
    # Perform CV rounds
    for i in tqdm.tqdm(range(config.rounds), desc="CV rounds"):
        cv_splits = StratifiedKFold(
            n_splits=config.splits,
            shuffle=True,
            random_state=i+10
        )
        
        # Process each fold
        for train_index, test_index in cv_splits.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Fit and evaluate model
            model.fit(X_train, y_train)
            extra_metrics_scores = _calculate_metrics(
                config.extra_metrics,
                extra_metrics_scores,
                model,
                X_test,
                y_test
            )

            # Calculate SHAP values if requested
            if config.calculate_shap:
                shap_values = _calc_shap(X_train, X_test, model)
                x_shap[test_index, :] += shap_values

    # Average SHAP values across rounds
    if config.calculate_shap:
        x_shap /= config.rounds

    logger.info(f"✓ Completed {config.rounds} CV rounds")
    return extra_metrics_scores, x_shap

def _bootstrap_evaluation(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    extra_metrics: Optional[list] = None,
    n_iterations: int = 100
) -> Dict:
    """
    Perform bootstrap evaluation for model evaluation.

    This method uses bootstrapping to create multiple training sets
    and evaluates the model on the out-of-bootstrap samples.
    """    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    
    # Perform bootstrap iterations
    for i in tqdm(range(n_iterations), desc="Bootstrap evaluation"):
        # Create bootstrap sample
        X_train_res, y_train_res = resample(X_train, y_train, random_state=i)
        
        # Train and evaluate model
        model_bootstrap = copy.deepcopy(model)
        model_bootstrap.fit(X_train_res, y_train_res)
        extra_metrics_scores = _calculate_metrics(
            extra_metrics,
            extra_metrics_scores,
            model_bootstrap,
            X_test,
            y_test
        )

    logger.info("✓ Bootstrap evaluation complete")
    return extra_metrics_scores

def _oob_evaluation(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    extra_metrics: Optional[list] = None,
    n_iterations: int = 100
) -> Dict:
    """
    Perform out-of-bag evaluation for model evaluation.

    This method evaluates the model using samples that were not used
    in the bootstrap training sets.
    """    
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}

    for i in tqdm(range(n_iterations), desc="OOB evaluation"):
        # Generate random indices for bootstrap sample
        rng = np.random.default_rng(i)
        indices = rng.choice(range(X.shape[0]), size=X.shape[0], replace=True)
        
        # Split into training and OOB sets
        X_train, y_train = X.iloc[indices, :], y[indices]
        oob_indices = list(set(range(X.shape[0])) - set(indices))
        X_test, y_test = X.iloc[oob_indices, :], y[oob_indices]

        # Train and evaluate model
        model_oob = copy.deepcopy(model)
        model_oob.fit(X_train, y_train)
        extra_metrics_scores = _calculate_metrics(
            extra_metrics,
            extra_metrics_scores,
            model_oob,
            X_test,
            y_test
        )

    logger.info("✓ OOB evaluation complete")
    return extra_metrics_scores

def _train_test_evaluation(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    processor: DataProcessor,
    extra_metrics: Optional[list] = None,
    n_splits: int = 100,
    test_size: float = 0.3
) -> Dict:
    """
    Perform train-test split evaluation with optional class balancing.

    This method creates multiple stratified train-test splits and
    optionally applies class balancing to the training data.
    """    
    extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    for i, (train_index, test_index) in enumerate(
        tqdm(sss.split(X, y), desc="Train-test evaluation", total=n_splits)
    ):
        # Apply class balancing if specified
        X_train, y_train = processor.class_balance_fnc(
            X.iloc[train_index],
            y[train_index],
            i=i
        )
        X_test, y_test = X.iloc[test_index], y[test_index]

        # Train and evaluate model
        model_tt = copy.deepcopy(model)
        model_tt.fit(X_train, y_train)
        extra_metrics_scores = _calculate_metrics(
            extra_metrics,
            extra_metrics_scores,
            model_tt,
            X_test,
            y_test
        )

    logger.info("✓ Train-test evaluation complete")
    return extra_metrics_scores