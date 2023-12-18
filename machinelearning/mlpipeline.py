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
from .mlestimator import MachineLearningEstimator

class MLPipelines(MachineLearningEstimator):

    def __init__(self, estimator, param_grid, label, csv_dir):
        ''' Class to perform machine learning pipelines 
            Inherits from MachineLearningEstimator class
            - estimator (sklearn estimator): estimator to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
        super().__init__(estimator, param_grid, label, csv_dir)

    def cross_validation(self, scoring='accuracy', cv=5) -> list:
        ''' Function to perform a simple cross validation
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
        returns:
            - scores (list): list of scores for each fold
        '''
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')
        
        scores = cross_val_score(self.estimator, self.X, self.y, cv=cv, scoring=scoring)
        print(f'Average {scoring}: {np.mean(scores)}')
        print(f'Standard deviation {scoring}: {np.std(scores)}')
        return scores

    def bootsrap(self, n_iter=100, test_size=0.2, optimizer='grid_search', random_iter=25, n_trials=100, cv=5, scoring='accuracy'):
        ''' Performs boostrap validation on a given estimator.
            - n_iter: number of iterations to perform boostrap validation
            - test_size: test size for each iteration
            - optimizer: 'grid_search' for GridSearchCV
                         'reandom_search' for RandomizedSearchCV
                         'bayesian_search' for optuna
            - random_iter: number of iterations for RandomizedSearchCV
            - n_trials: number of trials for optuna
            - cv: number of folds for cross-validation
            - scoring: scoring metric
    
        returns:
            - eval_metrics (list): list of evaluation metrics for each iteration
        '''                
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

        eval_metrics = []

        for i in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=i)
            if self.param_grid is None or self.param_grid == {}:
                self.estimator.fit(X_train, y_train)
            else:
                if optimizer == 'grid_search':
                    self.grid_search(X_train, y_train, scoring=scoring, cv=cv, verbose=False)
                elif optimizer == 'random_search':
                    self.random_search(X_train, y_train, scoring=scoring, cv=cv, n_iter=random_iter, verbose=False)
                elif optimizer == 'bayesian_search':
                    self.bayesian_search(X_train, y_train, scoring=scoring, direction='maximize', cv=cv, n_trials=n_trials, verbose=False)
                    self.best_estimator.fit(X_train, y_train)
                else:
                    raise ValueError(f'Invalid optimizer: {optimizer}. Select one of the following: grid_search, bayesian_search')
            
            y_pred = self.best_estimator.predict(X_test)
            eval_metrics.append(get_scorer(scoring)._score_func(y_test, y_pred))

        print(f'Average {scoring}: {np.mean(eval_metrics)}')
        print(f'Standard deviation {scoring}: {np.std(eval_metrics)}')
        return eval_metrics

    def nested_cross_validation(self, inner_scoring='accuracy', outer_scoring='accuracy',
                                inner_splits=3, outer_splits=5, optimizer='grid_search', 
                                n_trials=100, n_iter=25, num_trials=10, n_jobs=-1, verbose=0):
        ''' 
        Function to perform nested cross-validation for a given model and dataset in order to 
        perform model selection.

            - optimizer (str): 'grid_search' (GridSearchCV) 
                            'random_search' (RandomizedSearchCV) 
                            'bayesian_search' (optuna)
            - n_trials (int): Number of trials for optuna
            - n_iter (int):  Number of iterations for RandomizedSearchCV
            - num_trials (int): Number of trials for the nested cross-validation
            - n_jobs (int): Number of jobs to run in parallel
            - verbose (int): Verbosity level

        returns:
            - clf (object): Best model
            - nested_scores (list): Nested cross-validation scores
        '''

        # Check if both inner and outer scoring metrics are valid 
        if inner_scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')
        if outer_scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        print(f'Performing nested cross-validation for {self.estimator.__class__.__name__}...')
                
        nested_scores = []
        for i in tqdm(range(num_trials)):
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)
            
            if optimizer == 'grid_search':
                clf = GridSearchCV(estimator=self.estimator, scoring=inner_scoring, 
                                param_grid=self.param_grid, cv=inner_cv, n_jobs=n_jobs, verbose=verbose)
            elif optimizer == 'random_search':
                clf = RandomizedSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                         param_distributions=self.param_grid, cv=inner_cv, n_jobs=n_jobs, 
                                        verbose=verbose, n_iter=n_iter)
            elif optimizer == 'bayesian_search':
                clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                                        param_distributions=self.param_grid, cv=inner_cv, n_jobs=n_jobs, 
                                                        verbose=verbose, n_trials=n_trials)
            else:
                raise Exception("Unsupported optimizer.")
            
            nested_score = cross_val_score(clf, X=self.X, y=self.y, cv=outer_cv, scoring=outer_scoring, n_jobs=n_jobs)
            nested_scores.append(nested_score)

        nested_scores = [item for sublist in nested_scores for item in sublist]
        
        return nested_scores