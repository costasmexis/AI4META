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
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'LogisticRegression': LogisticRegression(),
            'XGBClassifier': XGBClassifier(),
            'GaussianNB': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'SVC': SVC(),
            'LGBMClassifier': LGBMClassifier(),
            'GaussianProcessClassifier': GaussianProcessClassifier(),
            'CatBoostClassifier': CatBoostClassifier()
        }
        
        if self.estimator is not None: # Checking if estimator is valid
            if self.name not in self.available_clfs.keys():
                raise ValueError(f'Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}')
        elif self.estimator is None: 
            print('There is no selected classifier.')
    
    @staticmethod
    def set_optuna_verbosity(level):
        """
        Set the verbosity level for Optuna.

        :param level: The verbosity level.
        :type level: int
        """
        optuna.logging.set_verbosity(level)  
        logging.getLogger("optuna").setLevel(level)

    def grid_search(self, X=None, y=None, scoring='matthews_corrcoef',
                        features_list=None, feat_num=None, feat_way='mrmr',
                        cv=5, verbose=True, return_model=False):
            """
            Perform grid search to find the best hyperparameters for the estimator.

            :param X: The input features, defaults to None
            :type X: array-like, shape (n_samples, n_features), optional
            :param y: The target values, defaults to None
            :type y: array-like, shape (n_samples,), optional
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
                    f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

            X = X or self.X
            y = y or self.y

            if features_list is not None:
                X = X[features_list]
            elif feat_num is not None:
                selected = self.feature_selection(X, y, feat_way, feat_num)
                X = X[selected]

            grid_search = GridSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv)
            grid_search.fit(X, y)
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.name = self.best_estimator.__class__.__name__
            self.best_estimator = grid_search.best_estimator_
            if verbose:
                print(f'Best parameters: {self.best_params}')
                print(f'Best {scoring}: {self.best_score}')

            if return_model:
                return self.best_estimator

    def random_search(self, X=None, y=None, scoring='matthews_corrcoef', 
                          features_list=None, cv=5, n_iter=100, verbose=True, 
                          return_model=False, feat_num=None, feat_way='mrmr'):
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
                    f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

            X = X or self.X
            y = y or self.y

            if features_list is not None:
                X = X[features_list]
            elif feat_num is not None:
                selected = self.feature_selection(X, y, feat_way, feat_num)
                X = X[selected]

            random_search = RandomizedSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv, n_iter=n_iter)
            random_search.fit(X, y)
            self.best_params = random_search.best_params_
            self.best_score = random_search.best_score_
            self.name = self.best_estimator.__class__.__name__
            self.best_estimator = random_search.best_estimator_

            if verbose:
                print(f'Best parameters: {self.best_params}')
                print(f'Best {scoring}: {self.best_score}')
            
            if return_model:
               return self.best_estimator

    def bayesian_search(self, X=None, y=None, scoring='matthews_corrcoef', features_list=None, rounds=10,
                            cv=5, direction='maximize', n_trials=100, estimator_name=None, evaluation='cv_simple',
                            feat_num=None, feat_way='mrmr', verbose=True, missing_values='median'):
        """
        Perform Bayesian hyperparameter search using Optuna.

        :param X: The input features, defaults to None.
        :type X: array-like, optional
        :param y: The target variable, defaults to None.
        :type y: array-like, optional
        :param scoring: The scoring metric to optimize, defaults to 'matthews_corrcoef'.
        :type scoring: str, optional
        :param features_list: The list of features to use, defaults to None.
        :type features_list: list, optional
        :param rounds: The number of rounds for cross-validation, defaults to 10.
        :type rounds: int, optional
        :param cv: The number of cross-validation folds, defaults to 5.
        :type cv: int, optional
        :param direction: The direction to maximize or minimize the scoring metric, defaults to 'maximize'.
        :type direction: str, optional
        :param n_trials: The number of trials for hyperparameter optimization, defaults to 100.
        :type n_trials: int, optional
        :param estimator_name: The name of the estimator, defaults to None.
        :type estimator_name: str, optional
        :param evaluation: The type of evaluation, defaults to 'cv_simple'.
        :type evaluation: str, optional
        :param feat_num: The number of features to select, defaults to None.
        :type feat_num: int, optional
        :param feat_way: The feature selection method, defaults to 'mrmr'.
        :type feat_way: str, optional
        :param verbose: Whether to print the best parameters and score, defaults to True.
        :type verbose: bool, optional
        :param missing_values: The method to handle missing values, defaults to 'median'.
        :type missing_values: str, optional
        :raises ValueError: If an invalid scoring metric is provided.
        :return: The best estimator or the best estimator and the evaluation results.
        :rtype: object or tuple
        """
        self.set_optuna_verbosity(logging.WARNING)

        grid = optuna_grid['ManualSearch']
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        else:
            scoring_function = getattr(metrics, scoring, None)
            scorer = make_scorer(scoring_function)

        X = X or self.X
        y = y or self.y

        if missing_values is not None:
            X = self.missing_values(X, method=missing_values)

        if features_list is not None:
            X = X[features_list]
        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        if estimator_name is None:
            estimator_name = self.name

        if evaluation == 'cv_rounds':
            columns = ['Hyperparameters', 'Score', 'Estimator', 'SEM']
            data_full_outer = pd.DataFrame(columns=columns)

            def c_v(i):
                cv_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)
                train_test_indices = list(cv_splits.split(X, y))
                local_data_full_outer = pd.DataFrame(columns=columns)
                for train_index, test_index in train_test_indices:
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    def objective(trial):
                        clf = grid[estimator_name](trial)
                        clf.fit(X_train, y_train)
                        score = scorer(clf, X_test, y_test)
                        return score

                    study = optuna.create_study(sampler=TPESampler(), direction=direction)
                    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
                    trial_scores = [trial.value for trial in study.trials]

                    new_row = {
                        'Hyperparameters': study.best_params,
                        'Score': study.best_value,
                        'Estimator': estimator_name,
                        'all_scores': trial_scores,
                        'SEM': np.std(trial_scores) / np.sqrt(len(trial_scores)),
                    }
                    new_row = pd.DataFrame([new_row])
                    local_data_full_outer = pd.concat([local_data_full_outer, new_row], ignore_index=True)
                return local_data_full_outer

            with threadpool_limits():
                list_dfs = Parallel(verbose=0)(delayed(c_v)(i) for i in range(rounds))

            data_full_outer = pd.concat(list_dfs, ignore_index=True)
            min_sem_index = data_full_outer['SEM'].idxmin()
            self.best_params = data_full_outer.loc[min_sem_index, 'Hyperparameters']
            self.best_score = data_full_outer.loc[min_sem_index, 'Score']
            best_clf = self.available_clfs[estimator_name]
            best_clf.set_params(**self.best_params)
            best_clf.fit(X, y)
            self.best_estimator = best_clf

        elif evaluation == 'cv_simple':
            def objective(trial):
                clf = grid[estimator_name](trial)
                final_score = cross_val_score(clf, X, y, scoring=scoring, cv=cv).mean()
                return final_score

            study = optuna.create_study(sampler=TPESampler(), direction=direction)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            self.best_params = study.best_params
            self.best_score = study.best_value
            best_clf = self.available_clfs[estimator_name]
            best_clf.set_params(**self.best_params)
            self.best_estimator = best_clf.fit(X, y)

            if verbose:
                print(f'For the {self.name} model: \nBest parameters: {self.best_params}\nBest {scoring}: {self.best_score}')

        if evaluation == 'cv_rounds':
            return self.best_estimator, data_full_outer
        else:
            return self.best_estimator

        