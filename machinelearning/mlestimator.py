import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, \
    GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import get_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from dataloader import DataLoader


class MachineLearningEstimator(DataLoader):
    def __init__(self, estimator, param_grid, label, csv_dir):
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
            'NaiveBayes': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'SVC': SVC()
        }

        self.bayesian_grid = {
            'RandomForestClassifier': lambda trial: RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                criterion='gini',  # or trial.suggest_categorical('criterion', ['gini', 'entropy'])
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                n_jobs=-1,
            ),
            'KNeighborsClassifier': lambda trial: KNeighborsClassifier(
                n_neighbors=trial.suggest_int('n_neighbors', 2, 15),
                weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
                algorithm=trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                p=trial.suggest_int('p', 1, 2),
                leaf_size=trial.suggest_int('leaf_size', 5, 50),
                n_jobs=-1
            ),
            'DecisionTreeClassifier': lambda trial: DecisionTreeClassifier(
                trial.suggest_categorical('criterion', ['gini', 'entropy']),
                splitter=trial.suggest_categorical('splitter', ['best', 'random']),
                max_depth=trial.suggest_int('max_depth', 1, 100),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_weight_fraction_leaf=trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            ),
            'SVC': lambda trial: SVC(
                C=trial.suggest_int('C', 1, 10),
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),
                probability=trial.suggest_categorical('probability', [True, False]),
                shrinking=trial.suggest_categorical('shrinking', [True, False]),
                decision_function_shape=trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
            ),
            'GradientBoostingClassifier': lambda trial: GradientBoostingClassifier(
                loss=trial.suggest_categorical('loss', ['log_loss', 'exponential']),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                criterion=trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
            ),
            'XGBClassifier': lambda trial: XGBClassifier(
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
                n_estimators=trial.suggest_int('n_estimators', 2, 200),
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                gamma=trial.suggest_float('gamma', 0.0, 0.5),
                subsample=trial.suggest_float('subsample', 0.1, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.1, 1.0),
                nthread=-1,
                verbosity=0
            ),
            'LinearDiscriminantAnalysis': lambda trial: LinearDiscriminantAnalysis(
                solver=trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                shrinkage=trial.suggest_float('shrinkage', 0.0, 1.0),
                n_components=trial.suggest_int('n_components', 1, 10)
            ),
            'LogisticRegression': lambda trial: LogisticRegression(
                penalty=trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
                C=trial.suggest_float('C', 0.1, 10.0),
                solver=trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                max_iter=trial.suggest_int('max_iter', 100, 1000),
                n_jobs=-1
            ),
            'NaiveBayes': lambda trial: GaussianNB(
                var_smoothing=trial.suggest_float('var_smoothing', 1e-9, 1e-5)
            )
        }
        
        # Check if the estimator is valid 
        if self.name not in self.available_clfs.keys():
            raise ValueError(f'Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}')
        
    def grid_search(self, X=None, y=None, scoring='accuracy', cv=5, verbose=True):
        ''' Function to perform a grid search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - verbose (bool): whether to print the results
        '''
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
        self.best_estimator = grid_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')

    def random_search(self, X=None, y=None, scoring='accuracy', cv=5, n_iter=100, verbose=True):
        ''' Function to perform a random search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - n_iter (int): number of iterations
            - verbose (bool): whether to print the results
        '''
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
        self.best_estimator = random_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')

    def bayesian_search():
        ''' TODO: To be implemented... '''
        pass



