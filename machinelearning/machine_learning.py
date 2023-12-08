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
    
    def nested_cross_validation(self, inner_splits=3, outer_splits=5,
                                inner_scoring='accuracy', outer_scoring='accuracy',
                                optimizer='grid_search', n_trials=100, num_trials=10, 
                                n_iter=25, verbose=0):
        ''' Function to perform a nested cross-validation
            - inner_splits (int): number of folds for inner cross-validation
            - outer_splits (int): number of folds for outer cross-validation
            - inner_scoring (str): scoring metric for inner cross-validation
            - outer_scoring (str): scoring metric for outer cross-validation
            - optimizer (str): 'grid_search' for GridSearchCV
                               'reandom_search' for RandomizedSearchCV
                               'bayesian_search' for optuna
            - n_trials (int): number of trials for optuna
            - num_trials (int): number of trials for the nested cross-validation
            - n_iter (int): number of iterations for RandomizedSearchCV
            - verbose (int): verbosity level
        returns:
            - nested_scores (list): list of scores for each fold
        '''
        # Check if both inner and outer scoring metrics are valid 
        if inner_scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')
        if outer_scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        nested_scores = []
        for i in tqdm(range(num_trials)):

            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)

            ''' TODO: UPDATE code below '''
            if optimizer == 'grid_search':
                pass
            elif optimizer == 'random_search':
                pass
            elif optimizer == 'bayesian_search':
                pass
            else:
                raise Exception("Unsupported optimizer.")

            ''' TODO: UPDATE cross_val_score below'''
            nested_score = cross_val_score(, X=self.X, y=self.y, scoring=outer_scoring, cv=outer_cv)
            nested_scores.append(list(nested_score))
        
        nested_scores = [item for sublist in nested_scores for item in sublist]
        return nested_scores


    def grid_search(self, X=None, y=None, scoring='accuracy', cv=5, verbose=True):
        ''' Function to perform a grid search
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
            - verbose (bool): whether to print the results
        '''
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

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
        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

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

    def bayesian_search(self, X=None, y=None, scoring='accuracy', direction='maximize', cv=5, n_trials=100, verbose=True):
        ''' Function to perform a bayesian search using Optuna 
            - X (array): features
            - y (array): target
            - scoring (str): scoring metric
            - direction (str): direction of the optimization
            - cv (int): number of folds for cross-validation
            - n_trials (int): number of trials
            - verbose (bool): whether to print the results
        '''

        if X is None and y is None:
            X = self.X 
            y = self.y

        if scoring not in sklearn.metrics.SCORERS.keys():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

        def create_model(trial):
            model = self.bayesian_grid[self.name](trial)
            return model

        def objective(trial):
            model = create_model(trial)
            model.fit(X, y)
            score = model.score(X, y)
            return score
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.best_estimator = self.estimator.set_params(**study.best_params)

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

    def bootsrap(self, n_iter=100, test_size=0.2, optimizer='grid_search', 
                 random_iter=25, n_trials=100, cv=5, scoring='accuracy'):
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
   

