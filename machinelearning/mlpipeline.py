import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
# from sklearn.metrics import matthews_corrcoef
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
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

from .mlestimator import MachineLearningEstimator
from .optuna_grid import optuna_grid
# from .Features_explanation import Features_explanation


class MLPipelines(MachineLearningEstimator):

    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        ''' Class to perform machine learning pipelines 
            Inherits from MachineLearningEstimator class
            - estimator (sklearn estimator): estimator to be used
            - param_grid (dict): hyperparameters to be tuned
            - label (str): name of the target column
            - csv_dir (str): path to the csv file
        '''
        super().__init__(label, csv_dir, estimator, param_grid)

    def cross_validation(self, scoring='matthews_corrcoef', cv=5) -> list:
        ''' Function to perform a simple cross validation
            - scoring (str): scoring metric
            - cv (int): number of folds for cross-validation
        returns:
            - scores (list): list of scores for each fold
        '''
        if scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        
        scores = cross_val_score(self.estimator, self.X, self.y, cv=cv, scoring=scoring)
        print(f'Average {scoring}: {np.mean(scores)}')
        print(f'Standard deviation {scoring}: {np.std(scores)}')
        return scores

    def bootsrap(self, n_iter=100, test_size=0.2, optimizer='grid_search', random_iter=25, n_trials=100, cv=5, scoring='matthews_corrcoef'):
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

    def nested_cross_validation(self, inner_scoring='matthews_corrcoef', outer_scoring='matthews_corrcoef',
                                inner_splits=3, outer_splits=5, optimizer='grid_search', 
                                n_trials=100, n_iter=25, num_trials=10, n_jobs=-1, verbose=0):
        """ Function to perform nested cross-validation for a given model and dataset in order to perform model selection

        Args:
            inner_scoring (str, optional): _description_. Defaults to matthews_corrcoef.
            outer_scoring (str, optional): _description_. Defaults to matthews_corrcoef.
            inner_splits (int, optional): _description_. Defaults to 3.
            outer_splits (int, optional): _description_. Defaults to 5.
            optimizer (str, optional): 'gird_search'   (GridSearchCV)
                                       'random_search' (RandomizedSearchCV))
                                       'bayesian_search' (Optuna) 
                                    Defaults to 'grid_search'.
            n_trials (int, optional): No. of trials for optuna. Defaults to 100.
            n_iter (int, optional): No. of iterations for RandomizedSearchCV. Defaults to 25.
            num_trials (int, optional): No. of trials for the nested cross-validation. Defaults to 10.
            n_jobs (int, optional): No. of jobs to run in parallel. Defaults to -1.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            nested_scores (list): nested scores
        """

        # Check if both inner and outer scoring metrics are valid 
        if inner_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        if outer_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

        print(f'Performing nested cross-validation for {self.estimator.__class__.__name__}...')
                
        nested_scores = []
        best_params_list = []
        for i in tqdm(range(num_trials)):
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)
            outer_scorer = get_scorer(outer_scoring)
            for train_index, test_index in outer_cv.split(self.X, self.y):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
        
                if optimizer == 'grid_search':
                    clf = GridSearchCV(estimator=self.estimator, scoring=inner_scoring, 
                                param_grid=self.param_grid, cv=inner_cv, n_jobs=n_jobs, verbose=verbose)
                elif optimizer == 'random_search':
                    clf = RandomizedSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                             param_distributions=self.param_grid, cv=inner_cv, n_jobs=n_jobs, 
                                             verbose=verbose, n_iter=n_iter)
                elif optimizer == 'bayesian_search':
                    clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                                        param_distributions=optuna_grid['NestedCV'][self.name], cv=inner_cv, n_jobs=n_jobs, 
                                                        verbose=verbose, n_trials=n_trials)
                else:
                    raise Exception("Unsupported optimizer.")        
                
                clf.fit(X_train, y_train)
                best_params_list.append(clf.best_params_)
                y_pred = clf.predict(X_test)
                nested_score = outer_scorer(clf, X_test, y_test)
                nested_scores.append(nested_score)


            # if optimizer == 'grid_search':
            #     clf = GridSearchCV(estimator=self.estimator, scoring=inner_scoring, 
            #                     param_grid=self.param_grid, cv=inner_cv, n_jobs=n_jobs, verbose=verbose)
            # elif optimizer == 'random_search':
            #     clf = RandomizedSearchCV(estimator=self.estimator, scoring=inner_scoring,
            #                              param_distributions=self.param_grid, cv=inner_cv, n_jobs=n_jobs, 
            #                             verbose=verbose, n_iter=n_iter)
            # elif optimizer == 'bayesian_search':
            #     clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
            #                                             param_distributions=optuna_grid['NestedCV'][self.name], cv=inner_cv, n_jobs=n_jobs, 
            #                                             verbose=verbose, n_trials=n_trials)
            # else:
            #     raise Exception("Unsupported optimizer.")
            
            # nested_score = cross_val_score(clf, X=self.X, y=self.y, cv=outer_cv, scoring=outer_scoring, n_jobs=n_jobs)
            # nested_scores.append(nested_score)

        # nested_scores = [item for sublist in nested_scores for item in sublist]
        
        return np.array(nested_scores),np.array(best_params_list)
    
    def model_selection(self,optimizer = 'grid_search', n_trials=100, n_iter=25, 
                        num_trials=10, score = 'matthews_corrcoef', exclude=None,
                        search_on=None, scores_df=False, alpha=1,
                        plot='box' , train_best=None, return_model=False):
        """
        Perform model selection using Nested Cross Validation.

        Parameters:
            optimizer (str, optional): The optimization method to use. Defaults to 'grid_search'.
                Options: 'grid_search', 'random_search', 'bayesian_search'.
            n_trials (int, optional): The number of trials for Bayesian optimization. Defaults to 100.
            n_iter (int, optional): The number of iterations for Random search. Defaults to 25.
            num_trials (int, optional): The number of cross-validation splits. Defaults to 10.
            score (str, optional): The scoring metric. Defaults to 'matthews_corrcoef'.
            exclude (list, optional): List of classifiers to exclude from selection. Defaults to None.
            search_on (list, optional): List of classifiers to include in selection. Defaults to None.
            scores_df (bool, optional): Whether to return a DataFrame of scores. Defaults to False.
            alpha (float, optional): The weight of standard deviation in the evaluation score. Defaults to 1.
            plot (str, optional): Type of plot for visualization ('box', 'violin', or None). Defaults to 'box'.
            train_best (str, optional): The method to retrain the best estimator on the whole dataset.
                Options: 'bayesian_search', 'grid_search', 'random_search', None. Defaults to None.
            return_model (bool, optional): Whether to return the best fitted estimator. Defaults to False.

        Returns:
            Fitted best estimator if return_model is True. Optionally returns a DataFrame of scores if scores_df is True.
            The return type depends on the flags `return_model` and `scores_df`:
                - If both are True, returns a tuple (best_estimator, scores_dataframe).
                - If only return_model is True, returns best_estimator.
                - If only scores_df is True, returns scores_dataframe.
                - Otherwise, returns None.
        """
        
        all_scores = []
        results = []
        
        if exclude is not None:
            exclude_classes = [classifier.__class__ for classifier in exclude]
        elif search_on is not None:
            classes = [classifier.__class__ for classifier in search_on]
            exclude_classes = [classifier.__class__ for classifier in self.available_clfs.values() if classifier.__class__ not in classes]
        else:
            exclude_classes = []

        clfs = [clf for clf in self.available_clfs.keys() if self.available_clfs[clf].__class__ not in exclude_classes]      
              
        for estimator in tqdm(clfs):

            # print(f'Performing nested cross-validation for {estimator}...')
            self.name = estimator
            self.estimator = self.available_clfs[estimator]
            scores_est,best_params_arr = self.nested_cross_validation(optimizer=optimizer, n_trials=n_trials, n_iter=n_iter,
                                          num_trials=num_trials, inner_scoring=score, outer_scoring=score)
            # scores_array = np.array([round(num, 4) for num in scores_est])
            print(type(scores_est))
            all_scores.append(scores_est)
            results.append({
            'Estimator': estimator,
            'Scores': scores_est,
            'Mean': np.mean(scores_est),
            'Max': np.max(scores_est),
            'Std': np.std(scores_est),
            'Median': np.median(scores_est),
            'Evaluated': np.mean(scores_est)-alpha*np.std(scores_est),
            'Best_Params': best_params_arr
        })
        
        if plot == 'box':
            plt.boxplot(all_scores, labels=clfs)
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(rotation=90)  
            plt.grid(True)
            plt.show()
        elif plot == 'violin':
            plt.violinplot(all_scores)
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(range(1, len(clfs) + 1), clfs, rotation=90)
            plt.grid(True)
            plt.show()
        elif plot == None: pass
        else: raise ValueError(f'The "{plot}" is not a valid option for plotting. Choose between "box", "violin" or None.')
        
        
        # best_estimator_name = max(results, key=lambda x: x['Mean Score'])['Estimator']
        self.name = max(results, key=lambda x: x['Evaluated'])['Estimator']
        self.estimator = self.available_clfs[self.name]
        # self.best_estimator = self.available_clfs[best_estimator_name]
        # self.estimator = self.available_clfs[best_estimator_name]
        # print(self.estimator)
        if train_best == 'bayesian_search':
            self.best_estimator = self.bayesian_search(cv=5, n_trials=n_trials, verbose=True,return_model=return_model)
        elif train_best == 'grid_search':
            self.best_estimator = self.grid_search(cv=5, verbose=True,return_model=return_model)
        elif train_best == 'random_search':
            self.best_estimator = self.random_search(cv=5, n_iter=25, verbose=True,return_model=return_model)
        elif train_best is None:
            print(f'Best estimator: {self.name}')
        else:   
            raise ValueError(f'Invalid type of best estimator train. Choose between "bayesian_search", "grid_search", "random_search" or None.')
        
        scores_dataframe = pd.DataFrame(results) if scores_df else None

        if return_model and scores_df:
            self.best_estimator = self.estimator.fit(self.X, self.y)
            return self.best_estimator, scores_dataframe
        elif return_model:
            self.best_estimator = self.estimator.fit(self.X, self.y)
            return self.best_estimator
        elif scores_df:
            return scores_dataframe

            