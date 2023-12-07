import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import get_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

    def cross_validation(self, X, y, scoring='accuracy', cv=5):
        ''' Function to perform a simple cross validation
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
        '''
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        scores = cross_val_score(self.estimator, X, y, cv=cv, scoring=scoring)
        print(f'Average {scoring}: {np.mean(scores)}')
        print(f'Standard deviation {scoring}: {np.std(scores)}')

    def grid_search(self, X, y, scoring='accuracy', cv=5, verbose=True):
        ''' Function to perform a grid search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - verbose (bool): whether to print the results
        '''
        grid_search = GridSearchCV(self.estimator, self.param_grid, scoring=scoring, cv=cv)
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')

    def random_search(self, X, y, scoring='accuracy', cv=5, n_iter=100, verbose=True):
        ''' Function to perform a random search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - n_iter (int): number of iterations
            - verbose (bool): whether to print the results
        '''
        random_search = RandomizedSearchCV(self.estimator, self.param_grid, scoring=scoring, \cv=cv, n_iter=n_iter)
        random_search.fit(X, y)
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_
        if verbose:
            print(f'Best parameters: {self.best_params}')
            print(f'Best {scoring}: {self.best_score}')

    def bayesian_search(self, X, y, scoring='accuracy', cv=5, n_trials=100, verbose=True):
        ''' TODO: Function to perform a bayesian search '''
        pass

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
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

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
                else:
                    raise ValueError(f'Invalid optimizer: {optimizer}. Select one of the following: grid_search, bayesian_search')
            
            y_pred = self.best_estimator.predict(X_test)
            eval_metrics.append(get_scorer(scoring)._score_func(y_test, y_pred))

        print(f'Average {scoring}: {np.mean(eval_metrics)}')
        print(f'Standard deviation {scoring}: {np.std(eval_metrics)}')
        return eval_metrics
   

# class BayesianOptimization(MachineLearningEstimator):
    
#     def __init__(self, X_train, y_train, X_test, y_test, 
#                  estimator, param_grid, label, csv_dir,
#                  scoring='accuracy', direction='maximize'):
        
#         super().__init__(estimator, param_grid, label, csv_dir)
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
        
#         self.scoring = scoring
#         self.direction = direction

#         if self.scoring not in sklearn.metrics.SCORERS.keys():
#             raise ValueError(f'Invalid scoring metric: {self.scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

#         self.available_clf = {
#             'RandomForestClassifier': RandomForestClassifier,
#             'KNeighborsClassifier': KNeighborsClassifier,
#             'DecisionTreeClassifier': DecisionTreeClassifier,
#             'SVC': SVC,
#             'GradientBoostingClassifier': GradientBoostingClassifier
#         }

#         if self.estimator in self.available_clf.values():
#             raise ValueError(f'Invalid estimator: {self.estimator}. Select one of the following: {list(self.available_clf.keys())}')
        
#         self.bayesian_clfs = {
#             'RandomForestClassifier': lambda trial: RandomForestClassifier(
#                 n_estimators=trial.suggest_int('n_estimators', 2, 200),
#                 criterion='gini',  # or trial.suggest_categorical('criterion', ['gini', 'entropy'])
#                 max_depth=trial.suggest_int('max_depth', 1, 50),
#                 min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
#                 min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
#                 bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
#                 n_jobs=-1,
#             ),
#             'KNeighborsClassifier': lambda trial: KNeighborsClassifier(
#                 n_neighbors=trial.suggest_int('n_neighbors', 2, 15),
#                 weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
#                 algorithm=trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
#                 p=trial.suggest_int('p', 1, 2),
#                 leaf_size=trial.suggest_int('leaf_size', 5, 50),
#                 n_jobs=-1
#             ),
#             'DecisionTreeClassifier': lambda trial: DecisionTreeClassifier(
#                 trial.suggest_categorical('criterion', ['gini', 'entropy']),
#                 splitter=trial.suggest_categorical('splitter', ['best', 'random']),
#                 max_depth=trial.suggest_int('max_depth', 1, 100),
#                 min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
#                 min_weight_fraction_leaf=trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
#             ),
#             'SVC': lambda trial: SVC(
#                 C=trial.suggest_int('C', 1, 10),
#                 kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),
#                 probability=trial.suggest_categorical('probability', [True, False]),
#                 shrinking=trial.suggest_categorical('shrinking', [True, False]),
#                 decision_function_shape=trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
#             ),
#             'GradientBoostingClassifier': lambda trial: GradientBoostingClassifier(
#                 loss=trial.suggest_categorical('loss', ['log_loss', 'exponential']),
#                 learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
#                 n_estimators=trial.suggest_int('n_estimators', 2, 200),
#                 criterion=trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
#                 max_depth=trial.suggest_int('max_depth', 1, 50),
#                 min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
#                 min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
#             )
#     }

#     def create_model(self, trial):
#         if self.estimator.__class__.__name__ in self.bayesian_clfs.keys():
#             model = self.bayesian_clfs[self.estimator](trial)
#         else:
#             raise ValueError('Classifier not supported')
#         return model
    
#     def objective(self, trial):
#         model = self.create_model(trial=trial)
#         model.fit(self.X_train, self.y_train)
#         y_pred = model.predict(self.X_test)
#         eval_metric = get_scorer(self.scoring)._score_func(self.y_test, y_pred)
#         return eval_metric

#     def run_optimization(self, n_trials=10):
#         self.clf = self.available_clf[self.estimator.__class__.__name__]

#         study = optuna.create_study(direction=self.direction)
#         study.optimize(self.objective, n_trials=n_trials)

#         if self.estimator.__class__.__name__ in self.bayesian_clfs:
#             self.model = self.available_clf[self.estimator.__class__.__name__](**study.best_params)
#         else:
#             raise ValueError('Classifier not supported')

#         return study

