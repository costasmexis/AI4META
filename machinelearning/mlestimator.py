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
    def __init__(self, estimator, param_grid, label, csv_dir):
        """Class to hold the machine learning estimator and related data
        Inherits from DataLoader class
        - estimator (sklearn estimator): estimator to be used
        - param_grid (dict): hyperparameters to be tuned
        - label (str): name of the target column
        - csv_dir (str): path to the csv file
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
        }

        # Check if the estimator is valid
        if self.name not in self.available_clfs.keys():
            raise ValueError(
                f"Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}"
            )

    def grid_search(self, X=None, y=None, scoring="accuracy", cv=5, verbose=True):
        """Function to perform a grid search

        Args:
            X (_type_, optional): Features. Defaults to None.
            y (_type_, optional): Target. Defaults to None.
            scoring (str, optional): Scoring metric. Defaults to "accuracy".
            cv (int, optional): No. of folds for cross-validation. Defaults to 5.
            verbose (bool, optional): Whether to print the results. Defaults to True.
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
        """Function to perform a random search

        Args:
            X (_type_, optional): Features. Defaults to None.
            y (_type_, optional): Target. Defaults to None.
            scoring (str, optional): Scoring metric. Defaults to "accuracy".
            cv (int, optional): No. of folds for cross-validation. Defaults to 5.
            n_iter (int, optional): No. of iterations. Defaults to 100.
            verbose (bool, optional): Whether to print the results. Defaults to True.
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
    ):  # , box=False):
        """Function to perform a bayesian search

        Args:
            X (_type_, optional): _description_. Defaults to None.
            y (_type_, optional): _description_. Defaults to None.
            scoring (str, optional): _description_. Defaults to 'accuracy'.
            cv (int, optional): _description_. Defaults to 5.
            direction (str, optional): _description_. Defaults to 'maximize'.
            n_trials (int, optional): _description_. Defaults to 100.
            verbose (bool, optional): _description_. Defaults to True.
            box (bool, optional): _description_. Defaults to False.
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
            clf = grid[self.name](trial)
            final_score = cross_val_score(clf, X, y, scoring=scoring, cv=cv).mean()
            return final_score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_estimator = grid[self.name](study.best_trial)

        if verbose:
            print(
                f"For the {self.name} model: \nBest parameters: {self.best_params}\nBest {scoring}: {self.best_score}"
            )
