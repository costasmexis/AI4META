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
import progressbar
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed,  parallel_backend

from dataloader import DataLoader
from .optuna_grid import optuna_grid
from optuna.samplers import TPESampler,RandomSampler
import logging
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics  
import shap
shap.initjs()

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

    def grid_search(self, X=None, y=None, scoring='matthews_corrcoef',
                    features_list=None, feat_num = None, feat_way = 'mrmr', 
                    cv=5, verbose=True, return_model=False):
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
        if features_list != None:
            X = X[features_list]
        elif feat_num != None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]
        else: pass
        
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

    def random_search(self, X=None, y=None, scoring='matthews_corrcoef', 
                      features_list=None,cv=5, n_iter=100, verbose=True, 
                      return_model=False, feat_num = None, feat_way = 'mrmr'):
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
        if features_list != None:
            X = X[features_list]
        elif feat_num != None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]
        else: pass

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

    def bayesian_search(self, X=None, y=None, scoring='matthews_corrcoef', features_list=None,rounds=10,
                        cv=5, direction='maximize', n_trials=100, estimator_name=None,evaluation='cv_simple',
                        feat_num = None, feat_way = 'mrmr', verbose=True, missing_values='median',calculate_shap=False):#, box=False):
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
        self.set_optuna_verbosity(logging.WARNING)

        grid = optuna_grid['ManualSearch']
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        else:
            scoring_function = getattr(metrics, scoring, None)
            scorer = make_scorer(scoring_function)
            
        if X is None and y is None:
            X = self.X
            y = self.y
            
        if missing_values != None:
            X = self.missing_values(X, method=missing_values)
            
        if features_list != None:
            X = X[features_list]
        elif feat_num != None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]
        else: pass

        if estimator_name == None:
            estimator_name = self.name 
        else: estimator_name = estimator_name
                              
                              
        if evaluation == 'cv_rounds':     
            # columns = ['best_hyperparameters','Score','Estimator','SEM']
            data_full_outer = pd.DataFrame()  
            def c_v(i):
                cv_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)
                train_test_indices = list(cv_splits.split(X, y))  
                local_data_full_outer = pd.DataFrame()  
                
                shaps=[]
                
                for train_index, test_index in train_test_indices:
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    def objective(trial):
                        clf = grid[estimator_name](trial)
                        clf.fit(X_train, y_train)
                        score = scorer(clf, X_test, y_test)
                        return score 
                    
                    study = optuna.create_study(sampler=TPESampler(),direction=direction)
                    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,n_jobs=-1)
                    trial_scores = [trial.value for trial in study.trials]
                    all_params = [trial.params for trial in study.trials]
                    
                    if calculate_shap:
                        best_params = study.best_params
                        best_model = self.available_clfs[estimator_name]
                        best_model.set_params(**best_params)
                        
                        try:
                            explainer = shap.Explainer(best_model, X_train)
                        except TypeError as e:
                            if "The passed model is not callable and cannot be analyzed directly with the given masker!" in str(e):
                                print("Switching to predict_proba due to compatibility issue with the model.")
                                explainer = shap.Explainer(lambda X: best_model.predict_proba(X), X_train)
                            else:
                                raise TypeError(e)
                        try:
                            shap_values = explainer(X_train)
                        except ValueError:
                            num_features = X_train.shape[1]
                            max_evals = 2 * num_features + 1
                            shap_values = explainer(X_train, max_evals=max_evals)    
                        
                        if len(shap_values.shape) == 3:
                            shap_values = shap_values[:,:,1]
                        else:
                            pass

                        new_row = {
                            'best_hyperparameters': study.best_params,
                            'Score': study.best_value,
                            'Estimator': estimator_name,
                            'all_scores': trial_scores,
                            'all_hyperparameters': all_params,
                            'SEM': np.std(trial_scores) / np.sqrt(len(trial_scores)),
                            'shap_values': shap_values.values
                        }
                        
                    else:
                        new_row = {
                            'best_hyperparameters': study.best_params,
                            'Score': study.best_value,
                            'Estimator': estimator_name,
                            'all_scores': trial_scores,
                            'all_hyperparameters': all_params,
                            'SEM': np.std(trial_scores) / np.sqrt(len(trial_scores))
                        }
                        
                    
                    new_row = pd.DataFrame([new_row])
                    local_data_full_outer = pd.concat([local_data_full_outer, new_row], ignore_index=True)
                    return local_data_full_outer
                
            with threadpool_limits():
                list_dfs = Parallel(verbose=0)(delayed(c_v)(i) for i in range(rounds))
            
            data_full_outer = pd.concat(list_dfs, ignore_index=True) 
            min_sem_index = data_full_outer['SEM'].idxmin()
            self.best_params = data_full_outer.loc[min_sem_index, 'best_hyperparameters']
            self.best_score = data_full_outer.loc[min_sem_index, 'Score']
            best_clf = self.available_clfs[estimator_name]
            best_clf.set_params(**self.best_params)
            best_clf.fit(X, y)
            self.best_estimator = best_clf
            
            ##Λαθος, μετραει μονο για 65 ενω θελω για 78. Χρησιμοποιησε τα ινδεχεσ
            
            if calculate_shap:
                all_shap_values = np.stack(data_full_outer['shap_values'].values)
                mean_shap_values = np.mean(all_shap_values, axis=0)
                self.shap_values = mean_shap_values
                data_full_outer.drop('shap_values', axis=1, inplace=True)
        
                                    
        elif evaluation == 'cv_simple':       
            def objective(trial):
                clf = grid[estimator_name](trial)
                final_score = cross_val_score(clf, X, y, scoring=scoring, cv=cv).mean()
                return final_score

            study = optuna.create_study(sampler=TPESampler(),direction=direction)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
       
            self.best_params = study.best_params
            self.best_score = study.best_value
            best_clf = self.available_clfs[estimator_name]
            best_clf.set_params(**self.best_params)
            self.best_estimator = best_clf.fit(X, y)
            
            # if verbose:
            #     print(f'For the {self.name} model: \nBest parameters: {self.best_params}\nBest {scoring}: {self.best_score}')
                
        if evaluation == 'cv_rounds':
            return  self.best_estimator, data_full_outer
        else:
            return self.best_estimator 
        

        