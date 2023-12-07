import sklearn
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from tqdm import tqdm

from .bayesian_opt import BayesianOptimization

def nested_cv(estimator, X, y, param_grid, 
              inner_splits=3, outer_splits=5, 
              inner_scoring='accuracy', 
              outer_scoring='accuracy',
              optimizer='grid_search', 
              n_trials=100, 
              num_trials=10, 
              n_iter=25, 
              verbose=0):
    ''' 
    Function to perform nested cross-validation for a given model and dataset.

        - optimizer (str): 'grid_search' (GridSearchCV) 
                           'random_search' (RandomizedSearchCV) 
                           'bayesian_search' (optuna)
        - n_trials (int): Number of trials for optuna
        - n_iter (int):  Number of iterations for RandomizedSearchCV
        - num_trials (int): Number of trials for the nested cross-validation

    returns:
        - clf (object): Best model
        - nested_scores (list): Nested cross-validation scores
    '''
    
    # Check if both inner and outer scoring metrics are valid 
    if inner_scoring not in sklearn.metrics.SCORERS.keys():
        raise ValueError(f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')
    if outer_scoring not in sklearn.metrics.SCORERS.keys():
        raise ValueError(f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.SCORERS.keys())}')

    print(f'Performing nested cross-validation for {estimator.__class__.__name__}...')
    
    nested_scores = []
    for i in tqdm(range(num_trials)):
        
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)
        
        if optimizer == 'grid_search':
            clf = GridSearchCV(estimator=estimator, scoring=inner_scoring, 
                            param_grid=param_grid, cv=inner_cv, n_jobs=-1, verbose=verbose)
        elif optimizer == 'random_search':
            clf = RandomizedSearchCV(estimator=estimator, scoring=inner_scoring, 
                                    param_distributions=param_grid, cv=inner_cv, n_jobs=-1, 
                                    verbose=verbose, n_iter=n_iter) 
        else:
            raise Exception("Unsupported optimizer.")
        
        nested_score = cross_val_score(clf, X=X, y=y, scoring=outer_scoring, cv=outer_cv)
        nested_scores.append(list(nested_score))

    nested_scores = [item for sublist in nested_scores for item in sublist]

    return clf, nested_scores
