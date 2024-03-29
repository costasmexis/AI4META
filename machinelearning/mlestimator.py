import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from dataloader import DataLoader
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
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

        if self.estimator is None:
            print('You have not provided an estimator.')
        elif self.estimator.__class__.__name__ not in self._available_clfs.keys():
            raise ValueError(
                f"Invalid estimator: {self.estimator.__class__.__name__}. Select one of the following: {list(self._available_clfs.keys())}"
            )

    def grid_search(self, X=None, y=None, scoring: str="matthews_corrcoef", cv: int=5, 
                    verbose: bool=True, return_model: bool=False):
        """ Performs a grid search

        :param X: Features vector, defaults to ``None``
        :type X: np.array, optional
        :param y: Target vector, defaults to ``None``
        :type y: np.array, optional
        :param scoring: Scoring metric, defaults to ``matthews_corrcoef``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param verbose: Verbose, defaults to ``True``
        :type verbose: bool, optional
        :param return_model: Return the best model fitted on X, y, defaults to ``False``
        :type return_model: bool, optional
        """        
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )
                        
        if X is None and y is None:
            X = self.X 
            y = self.y
          
        grid_search = GridSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv)
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        if verbose:
            print(f"Best parameters: {self.best_params}")
            print(f"Best {scoring}: {self.best_score}")
        if return_model:
            return self.best_estimator.fit(self.X, self.y )

    def random_search(
        self, X=None, y=None, scoring="matthews_corrcoef", cv=5, n_iter=100, verbose=True, return_model=False
    ):
        """ Performs a random search

        :param X: Features vector, defaults to ``None``
        :type X: np.array, optional
        :param y: Target vector, defaults to ``None``
        :type y: np.array, optional
        :param scoring: Scoring metric, defaults to ``matthews_corrcoef``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param n_iter: Number of interations for Random Search, defaults to 100
        :type n_iter: int, optional
        :param verbose: Verbose, defaults to ``True``
        :type verbose: bool, optional
        :param return_model: Return the best model fitted on X, y, defaults to ``False``
        :type return_model: bool, optional
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
        if return_model:
            return self.best_estimator.fit(self.X, self.y)

    def bayesian_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        cv=5,
        direction="maximize",
        n_trials=100,
        verbose=True,
        return_model=False
    ):  
        """ Performs a bayesian search

        :param X: Features vector, defaults to ``None``
        :type X: np.array, optional
        :param y: Target vector, defaults to ``None``
        :type y: np.array, optional
        :param scoring: Scoring metric, defaults to ``matthews_corrcoef``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param direction: *Minimize* or *Maximize* the scoring metric, defaults to ``maximize``
        :type direction: str, optional
        :param n_trials: Number of trials for bayesian search, defaults to 100
        :type n_trials: int, optional
        :param verbose: Verbose, defaults to ``True``
        :type verbose: bool, optional
        :param return_model: Return the best model fitted on X, y, defaults to ``False``
        :type return_model: bool, optional
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
        
        study = optuna.create_study(sampler=TPESampler(), direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_estimator = self.estimator.__class__(**self.best_params).fit(self.X, self.y)

        if verbose:
            print(
                f"For the {self.estimator.__class__.__name__} model: \nBest parameters: {self.best_params}\nBest {scoring}: {self.best_score}"
            )
        if return_model:
            return self.best_estimator.fit(self.X, self.y)

    def cross_validation(self, scoring: str = "matthews_corrcoef", cv: int = 5) -> list:
        """ Performs cross validation on a given estimator

        :param scoring: Scoring metric, defaults to ``matthews_corrcoef``
        :type scoring: str, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :return: List of scores for each fold
        :rtype: list
        """        
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        scores = cross_val_score(self.estimator, self.X, self.y, cv=cv, scoring=scoring)
        print(f"Average {scoring}: {np.mean(scores)}")
        print(f"Standard deviation {scoring}: {np.std(scores)}")
        return scores

    def bootstrap(
        self,
        n_iter=10,
        test_size=0.2,
        optimizer="grid",
        random_iter=25,
        n_trials=100,
        cv=5,
        scoring="matthews_corrcoef",
        verbose=False
    ):
        """Performs boostrap validation on a given estimator.

        :param n_iter: Number of iterations to perform bootstrap validation, defaults to 100
        :type n_iter: int, optional
        :param test_size: Test size for each iteration, defaults to 0.2
        :type test_size: float, optional
        :param optimizer: Method to use for hyperparameter optimization, defaults to ``grid``
        :type optimizer: str, optional
        :param random_iter: Number of iterations for ``RandomSearchCV``, defaults to 25
        :type random_iter: int, optional
        :param n_trials: Number of trials for ``Optuna``, defaults to 100
        :type n_trials: int, optional
        :param cv: Number of folds for Cross Validation, defaults to 5
        :type cv: int, optional
        :param scoring: Scoring metric, defaults to ``matthews_corrcoef``
        :type scoring: str, optional
        :return: List of evaluation metrics for each iteration
        :rtype: list
        """        
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )

        eval_metrics = []
        for i in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=i)

            if (self.param_grid is None or self.param_grid == {}) or (optimizer=='evaluation'):
                self.best_estimator.fit(X_train, y_train)
            else:
                if optimizer == "grid":
                    self.grid_search(X_train, y_train, scoring=scoring, cv=cv, verbose=verbose)
                elif optimizer == "random":
                    self.random_search(X_train, y_train, scoring=scoring, cv=cv, n_iter=random_iter, verbose=verbose)
                elif optimizer == "bayesian":
                    self.bayesian_search(X_train, y_train, scoring=scoring, direction="maximize", cv=cv, n_trials=n_trials, verbose=verbose)
                    self.best_estimator.fit(X_train, y_train)
                else:
                    raise ValueError(f"Invalid optimizer: {optimizer}. Select one of the following: grid, random, bayesian")

            y_pred = self.best_estimator.predict(X_test)
            eval_metrics.append(get_scorer(scoring)._score_func(y_test, y_pred))

        print(f"Average {scoring}: {np.mean(eval_metrics)}")
        print(f"Standard deviation {scoring}: {np.std(eval_metrics)}")
        return eval_metrics
