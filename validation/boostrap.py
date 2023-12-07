import sklearn
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

from .bayesian_opt import BayesianOptimization

def boostrap(
        estimator,
        X,
        y,
        param_grid: dict = None,
        n_iterations=100,
        test_size=0.2,
        optimizer='grid_search',
        n_trials=100,
        scoring = 'accuracy'):
    
    ''' Performs boostrap validation on a given estimator.
        - estimator: sklearn estimator
        - X: features
        - y: target
        - param_grid: hyperparameter grid to search. If 'None' the estimator will be trained with the given/default hyperparameters.
        - n_iterations: number of iterations to perform boostrap validation
        - test_size: test size for each iteration
        - optimizer: 'grid_search' for GridSearchCV
                    'bayesian_search' for optuna
        - n_trials: number of trials for optuna
        - scoring: scoring metric
  
      returns:
        - evaluation_metrics (list): list of evaluation metrics for each iteration
    '''
    
    if scoring not in sklearn.metrics.SCORERS.keys():
        raise ValueError(f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

    evaluation_metrics = []

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        if param_grid == None:
            if optimizer == 'bayesian_search':
                bayopt = BayesianOptimization(X_train, y_train, X_test, y_test, scoring=scoring)
                bayopt.run_optimization(classifier=estimator, n_trials=n_trials)
                estimator = bayopt.model
                estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train)
        else:
            grid = GridSearchCV(estimator, param_grid, scoring=scoring, cv=5)
            grid.fit(X_train, y_train)
            estimator = grid.best_estimator_

        y_pred = estimator.predict(X_test)
        evaluation_metrics.append(get_scorer(scoring)._score_func(y_test, y_pred))

    print(f'Average {scoring}: {np.mean(evaluation_metrics)}')
    print(f'Standard deviation {scoring}: {np.std(evaluation_metrics)}')

    return evaluation_metrics