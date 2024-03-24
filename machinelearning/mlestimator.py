import numpy as np
import optuna
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# from sklearn.metrics import matthews_corrcoef
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
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier

from dataloader import DataLoader
from .optuna_grid import optuna_grid
from optuna.samplers import TPESampler,RandomSampler
import logging
# from logging_levels import add_log_level

class MachineLearningEstimator(DataLoader):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        ''' Class to hold the machine learning estimator and related data 
            Inherits from DataLoader class
            - estimator (sklearn estimator): estimator to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
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
        
        if self.estimator is not None:
            # Check if the estimator is valid
            if self.name not in self.available_clfs.keys():
                raise ValueError(f'Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}')
        else: print(f'There is no selected classifier.')
        
    def set_optuna_verbosity(self, level):
        """Adjust Optuna's verbosity level."""
        optuna.logging.set_verbosity(level)  
        logging.getLogger("optuna").setLevel(level)

    def grid_search(self, X=None, y=None, scoring='matthews_corrcoef', cv=5, verbose=True, return_model=False):
        """ Function to perform a grid search
        Args:
            X (array, optional): Features. Defaults to None.
            y (array, optional): Target. Defaults to None.
            scoring (str, optional): Scoring metric. Defaults to 'matthews_corrcoef'.
            cv (int, optional): No. of folds for cross-validation. Defaults to 5.
            verbose (bool, optional): Whether to print the results. Defaults to True.
        """
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        
        if X is None and y is None:
            X = self.X 
            y = self.y
        
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

    def random_search(self, X=None, y=None, scoring='matthews_corrcoef', cv=5, n_iter=100, verbose=True, return_model=False):
        """ Function to perform a random search
        Args:
            X (array, optional): Features. Defaults to None.
            y (array, optional): Target. Defaults to None.
            scoring (str, optional): Scoring metric. Defaults to 'matthews_corrcoef'.
            cv (int, optional): No. of folds for cross-validation. Defaults to 5.
            n_iter (int, optional): No. of iterations. Defaults to 100.
            verbose (bool, optional): Whether to print the results. Defaults to True.
        """
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

        if X is None and y is None:
            X = self.X 
            y = self.y
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

    def bayesian_search(self, X=None, y=None, scoring='matthews_corrcoef', 
                        cv=5, direction='maximize', n_trials=100, estimator_name=None,
                        verbose=True, return_model=False):#, box=False):
        """ Function to perform a bayesian search

        Args:
            X (_type_, optional): _description_. Defaults to None.
            y (_type_, optional): _description_. Defaults to None.
            scoring (str, optional): _description_. Defaults to 'matthews_corrcoef'.
            cv (int, optional): _description_. Defaults to 5.
            direction (str, optional): _description_. Defaults to 'maximize'.
            n_trials (int, optional): _description_. Defaults to 100.
            verbose (bool, optional): _description_. Defaults to True.
            box (bool, optional): _description_. Defaults to False.
        Returns:
            _type_: _description_
        """
        self.set_optuna_verbosity(logging.ERROR)

        grid = optuna_grid['ManualSearch']
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
            
        if X is None and y is None:
            X = self.X
            y = self.y
        if estimator_name == None:
            estimator_name = self.name 
        else: estimator_name = estimator_name
                              
        def objective(trial):
            clf = grid[estimator_name](trial)
            final_score = cross_val_score(clf, X, y, scoring=scoring, cv=cv).mean()
            return final_score

        # if set_sampler == 'TPESampler':
        study = optuna.create_study(sampler=TPESampler(),direction=direction)
        # elif set_sampler == 'RandomSampler':
        #     study = optuna.create_study(sampler=RandomSampler(),direction=direction)
        # elif set_sampler == 'CmaEsSampler':
        #     study = optuna.create_study(sampler=CmaEsSampler(),direction=direction)
        # elif set_sampler == None:
        #     study = optuna.create_study(direction=direction)
        # else: raise ValueError('Invalid sampler, Choose between TPESampler, RandomSampler or None')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_estimator = grid[estimator_name](study.best_trial)
        self.name = self.best_estimator.__class__.__name__
        
        self.best_estimator.fit(X, y) #fit in all X,y data

        if verbose:
            print(f'For the {self.name} model: \nBest parameters: {self.best_params}\nBest {scoring}: {self.best_score}')

        if return_model:
            return self.best_estimator #returns the best fitted estimator
            
        
        # if box:
        #     best_scores = [trial.value for trial in study.trials if trial.value is not None]
        #     plt.style.use('seaborn-whitegrid')
        #     plt.boxplot(best_scores, widths=0.75, whis=2)
        #     plt.ylim(0, 1)
        #     plt.title(f"Cross-Validation Scores Across Trials for {self.name}")
        #     plt.ylabel('Scores')
        #     plt.xlabel(f'{cv}-Fold Cross-Validation')
        #     plt.show()

        