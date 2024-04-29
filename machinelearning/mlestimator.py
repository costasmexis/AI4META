import logging

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import progressbar
import sklearn
import sklearn.metrics as metrics
from catboost import CatBoostClassifier
from joblib import Parallel, delayed, parallel_backend
from lightgbm import LGBMClassifier
from optuna.samplers import RandomSampler, TPESampler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, make_scorer
import sklearn.metrics
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from xgboost import XGBClassifier

from dataloader import DataLoader

from .optuna_grid import optuna_grid


class MachineLearningEstimator(DataLoader):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        """
        Initialize the MLEstimator object.

        :param label: The label for the estimator.
        :type label: str
        :param csv_dir: The directory containing the CSV files.
        :type csv_dir: str
        :param estimator: The estimator object to use, defaults to None.
        :type estimator: object, optional
        :param param_grid: The parameter grid for hyperparameter tuning, defaults to None.
        :type param_grid: dict, optional
        """

        super().__init__(label, csv_dir)
        self.estimator = estimator
        self.name = estimator.__class__.__name__
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None

        self.available_clfs = {
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
            "LogisticRegression": LogisticRegression(),
            "XGBClassifier": XGBClassifier(),
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "SVC": SVC(),
            "LGBMClassifier": LGBMClassifier(),
            "GaussianProcessClassifier": GaussianProcessClassifier(),
            "CatBoostClassifier": CatBoostClassifier(),
        }

        if self.estimator is not None:  # Checking if estimator is valid
            if self.name not in self.available_clfs.keys():
                raise ValueError(
                    f"Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}"
                )
        elif self.estimator is None:
            print("There is no selected classifier.")

    @staticmethod
    def set_optuna_verbosity(level):
        """
        Set the verbosity level for Optuna.

        :param level: The verbosity level.
        :type level: int
        """
        optuna.logging.set_verbosity(level)
        logging.getLogger("optuna").setLevel(level)

    def grid_search(
        self,
        X=None,
        y=None,
        estimator=None,
        parameter_grid=None,
        scoring="matthews_corrcoef",
        features_list=None,
        feat_num=None,
        feat_way="mrmr",
        cv=5,
        verbose=True,
        return_model=False,
    ):
        """
        Perform grid search to find the best hyperparameters for the estimator.

        :param X: The input features, defaults to None
        :type X: array-like, shape (n_samples, n_features), optional
        :param y: The target values, defaults to None
        :type y: array-like, shape (n_samples,), optional
        :param estimator: The estimator object. If None self.estimator is selected, defaults to None.
        :type estimator: object, optional
        :param parameter_grid: The parameter grid for hyperparameter tuning, defaults to None
        :type parameter_grid: dict, optional
        :param scoring: The scoring metric to optimize, defaults to 'matthews_corrcoef'
        :type scoring: str, optional
        :param features_list: List of feature names to use for grid search, defaults to None
        :type features_list: list, optional
        :param feat_num: Number of features to select using feature selection method, defaults to None
        :type feat_num: int, optional
        :param feat_way: The feature selection method to use, defaults to 'mrmr'
        :type feat_way: str, optional
        :param cv: The number of cross-validation folds, defaults to 5
        :type cv: int, optional
        :param verbose: Whether to print the best parameters and score, defaults to True
        :type verbose: bool, optional
        :param return_model: Whether to return the best estimator, defaults to False
        :type return_model: bool, optional
        :return: The best estimator if `return_model` is True, otherwise None
        :rtype: estimator object or None
        """

        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        X = X or self.X
        y = y or self.y

        if len(X) != len(y):
            raise ValueError("The length of X and y must be equal.")

        if features_list is not None:
            X = X[features_list]
        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        if estimator is not None:
            self.estimator = estimator
            self.name = self.estimator.__class__.__name__
            self.param_grid = parameter_grid

        grid_search = GridSearchCV(
            self.estimator, self.param_grid, scoring=scoring, cv=cv
        )
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.name = self.best_estimator.__class__.__name__
        self.best_estimator = grid_search.best_estimator_
        if verbose:
            print(f"Best parameters: {self.best_params}")
            print(f"Best {scoring}: {self.best_score}")

        if return_model:
            return self.best_estimator

    def random_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        estimator=None,
        parameter_grid=None,
        features_list=None,
        cv=5,
        n_iter=100,
        verbose=True,
        return_model=False,
        feat_num=None,
        feat_way="mrmr",
    ):
        """
        Perform a random search for hyperparameter tuning.

        :param X: The input features, defaults to None
        :type X: array-like, shape (n_samples, n_features), optional
        :param y: The target variable, defaults to None
        :type y: array-like, shape (n_samples,), optional
        :param scoring: The scoring metric to optimize, defaults to 'matthews_corrcoef'
        :type scoring: str, optional
        :param features_list: A list of feature names to select from X, defaults to None
        :type features_list: list, optional
        :param cv: The number of cross-validation folds, defaults to 5
        :type cv: int, optional
        :param n_iter: The number of parameter settings that are sampled, defaults to 100
        :type n_iter: int, optional
        :param verbose: Whether to print the best parameters and score, defaults to True
        :type verbose: bool, optional
        :param return_model: Whether to return the best estimator, defaults to False
        :type return_model: bool, optional
        :param feat_num: The number of features to select using feature selection, defaults to None
        :type feat_num: int, optional
        :param feat_way: The feature selection method, defaults to 'mrmr'
        :type feat_way: str, optional
        :raises ValueError: If an invalid scoring metric is provided
        :return: The best estimator if return_model is True, otherwise None
        :rtype: estimator object or None
        """
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        X = X or self.X
        y = y or self.y

        if len(X) != len(y):
            raise ValueError("The length of X and y must be equal.")

        if features_list is not None:
            X = X[features_list]
        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        if estimator is not None:
            self.estimator = estimator
            self.name = self.estimator.__class__.__name__
            self.param_grid = parameter_grid

        random_search = RandomizedSearchCV(
            self.estimator, self.param_grid, scoring=scoring, cv=cv, n_iter=n_iter
        )
        random_search.fit(X, y)
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.name = self.best_estimator.__class__.__name__
        self.best_estimator = random_search.best_estimator_

        if verbose:
            print(f"Best parameters: {self.best_params}")
            print(f"Best {scoring}: {self.best_score}")

        if return_model:
            return self.best_estimator

    def bayesian_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        estimator=None,
        direction="maximize",
        n_trials=100,
        features_list=None,
        feat_num=None,
        feat_way="mrmr",
        cv=5,
        verbose=True,
        return_model=False,
    ):
        """
        Perform Bayesian hyperparameter search using Optuna.

        :param X: The input features, defaults to None
        :type X: array-like, shape (n_samples, n_features), optional
        :param y: The target values, defaults to None
        :type y: array-like, shape (n_samples,), optional
        :param scoring: The scoring metric to optimize, defaults to 'matthews_corrcoef'
        :type scoring: str, optional
        :param estimator: The base estimator to use, defaults to None
        :type estimator: estimator object, optional
        :param direction: The direction to optimize the scoring metric, defaults to 'maximize'
        :type direction: str, optional
        :param n_trials: The number of trials for the hyperparameter search, defaults to 100
        :type n_trials: int, optional
        :param features_list: The list of feature names to select, defaults to None
        :type features_list: list, optional
        :param feat_num: The number of features to select, defaults to None
        :type feat_num: int, optional
        :param feat_way: The feature selection method, defaults to 'mrmr'
        :type feat_way: str, optional
        :param cv: The number of cross-validation folds, defaults to 5
        :type cv: int, optional
        :param verbose: Whether to print the best parameters and score, defaults to True
        :type verbose: bool, optional
        :param return_model: Whether to return the best estimator, defaults to False
        :type return_model: bool, optional
        :return: The best estimator if `return_model` is True, otherwise None
        :rtype: estimator object or None
        """

        self.set_optuna_verbosity(logging.WARNING)

        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        X = X or self.X
        y = y or self.y

        if len(X) != len(y):
            raise ValueError("The length of X and y must be equal.")

        if features_list is not None:
            X = X[features_list]
        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        if estimator is not None:
            self.estimator = estimator
            self.name = self.estimator.__class__.__name__

        def objective(trial):
            cls = optuna_grid["ManualSearch"][self.name](trial)
            score = cross_val_score(cls, X, y, scoring=scoring, cv=cv).mean()
            return score

        study = optuna.create_study(sampler=TPESampler(), direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_estimator = self.available_clfs[self.name].set_params(
            **self.best_params
        )
        self.best_estimator.fit(X, y)

        if verbose:
            print(f"Best parameters: {self.best_params}")
            print(f"Best {scoring}: {self.best_score}")

        if return_model:
            return self.best_estimator
