import numpy as np
import optuna
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
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
from tqdm import tqdm
from xgboost import XGBClassifier

from dataloader import DataLoader
from .optuna_grid import optuna_grid


class MachineLearningEstimator(DataLoader):
    """ Class to hold the machine learning estimator information and related data 

    :param estimator: Estimator to be used
    :type estimator: sklearn estimator
    :param param_grid: Hyperparameters grid to be searched
    :type param_grid: dict
    :param label: Name of target column
    :type label: str
    :param csv_dir: Path to the csv file
    :type csv_dir: str
    """        
    def __init__(self, estimator, param_grid: dict, label: str, csv_dir: str):

        super().__init__(label, csv_dir)
        self.estimator = estimator
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
        }

        # Check if estimator is NoneType
        if self.estimator is None:
            print('You have not provided an estimator.')
        elif self.estimator.__class__.__name__ not in self._available_clfs.keys():
            raise ValueError(
                f"Invalid estimator: {self.estimator.__class__.__name__}. Select one of the following: {list(self._available_clfs.keys())}"
            )

    def grid_search(self, X=None, y=None, scoring="accuracy", cv=5, verbose=True):
        """ Performs a grid search

        :param X: Features vector, defaults to ``None``
        :type X: np.array, optional
        :param y: Target vector, defaults to ``None``
        :type y: np.array, optional
        :param scoring: Scoring metric, defaults to ``accuracy``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param verbose: Verbose, defaults to ``True``
        :type verbose: bool, optional
        """        
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        if X is None and y is None:
            X = self.X
            y = self.y

        grid_search = GridSearchCV(
            self.estimator, self.param_grid, scoring=scoring, cv=cv
        )
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        if verbose:
            print(f"Best parameters: {self.best_params}")
            print(f"Best {scoring}: {self.best_score}")

    def random_search(
        self, X=None, y=None, scoring="accuracy", cv=5, n_iter=100, verbose=True
    ):
        """ Performs a random search

        :param X: Features vector, defaults to ``None``
        :type X: np.array, optional
        :param y: Target vector, defaults to ``None``
        :type y: np.array, optional
        :param scoring: Scoring metric, defaults to ``accuracy``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param n_iter: Number of interations for Random Search, defaults to 100
        :type n_iter: int, optional
        :param verbose: Verbose, defaults to ``True``
        :type verbose: bool, optional
        """        
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        if X is None and y is None:
            X = self.X
            y = self.y
        random_search = RandomizedSearchCV(
            self.estimator, self.param_grid, scoring=scoring, cv=cv, n_iter=n_iter
        )
        random_search.fit(X, y)
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_
        if verbose:
            print(f"Best parameters: {self.best_params}")
            print(f"Best {scoring}: {self.best_score}")

    def bayesian_search(
        self,
        X=None,
        y=None,
        scoring="accuracy",
        cv=5,
        direction="maximize",
        n_trials=100,
        verbose=True,
    ):  
        """ Performs a bayesian search

        :param X: Features vector, defaults to ``None``
        :type X: np.array, optional
        :param y: Target vector, defaults to ``None``
        :type y: np.array, optional
        :param scoring: Scoring metric, defaults to ``accuracy``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param direction: *Minimize* or *Maximize* the scoring metric, defaults to ``maximize``
        :type direction: str, optional
        :param n_trials: Number of trials for bayesian search, defaults to 100
        :type n_trials: int, optional
        :param verbose: Verbose, defaults to ``True``
        :type verbose: bool, optional
        """        
        grid = optuna_grid["ManualSearch"]
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        if X is None and y is None:
            X = self.X
            y = self.y

        def objective(trial):
            clf = grid[self.estimator.__class__.__name__](trial)
            final_score = cross_val_score(clf, X, y, scoring=scoring, cv=cv).mean()
            return final_score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_estimator = self.estimator.__class__(**self.best_params).fit(self.X, self.y)

        if verbose:
            print(
                f"For the {self.estimator.__class__.__name__} model: \nBest parameters: {self.best_params}\nBest {scoring}: {self.best_score}"
            )
