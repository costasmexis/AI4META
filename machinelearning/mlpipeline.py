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
# from joblib import Parallel, delayed
from dataloader import DataLoader

from .mlestimator import MachineLearningEstimator
from .optuna_grid import optuna_grid

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from mrmr import mrmr_classif
import logging
# from numba import jit, prange
# from .Features_explanation import Features_explanation
from multiprocessing import Pool
from itertools import chain

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
        self.set_optuna_verbosity(logging.ERROR)

    def set_optuna_verbosity(self, level):
        """Adjust Optuna's verbosity level."""
        optuna.logging.set_verbosity(level)  
        logging.getLogger("optuna").setLevel(level) 

    
    # @jit(parallel=True)
    def inner_loop(self, X_train=None, X_test=None, y_train=None, y_test=None):
        nested_scores = []
        
        optimizer = self.params['optimizer']
        inner_scoring = self.params['inner_scoring']
        inner_cv = self.params['inner_cv']
        n_jobs = self.params['n_jobs']
        verbose = self.params['verbose']
        n_iter = self.params['n_iter']
        n_trials_ncv = self.params['n_trials_ncv']
        outer_scoring = self.params['outer_scoring']
        outer_scorer = get_scorer(outer_scoring)

        if optimizer == 'grid_search':
            clf = GridSearchCV(estimator=self.estimator, scoring=inner_scoring, 
                              param_grid=self.param_grid, cv=inner_cv, n_jobs=n_jobs, verbose=verbose)
        elif optimizer == 'random_search':
                    clf = RandomizedSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                             param_distributions=self.param_grid, cv=inner_cv, n_jobs=n_jobs, 
                                             verbose=verbose, n_iter=n_iter)
        elif optimizer == 'bayesian_search':
            clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                                    param_distributions=optuna_grid['NestedCV'][self.name],
                                                    cv=inner_cv, n_jobs=n_jobs, 
                                                    verbose=verbose, n_trials=n_trials_ncv)
        else:
            raise Exception("Unsupported optimizer.")   
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        nested_score = outer_scorer(clf, X_test, y_test)
        nested_scores.append(nested_score)
        return nested_scores 
    
    
    # @jit(parallel=True)
    def clf_app(self, X_train_selected, X_test_selected, y_train, y_test,num_feature):
        nested_scores = []
        classifiers_list = []
        number_of_features = []
        way_of_selection = []
        estimator_list = []
        percentile_list = []
        lists = [nested_scores, classifiers_list, number_of_features, way_of_selection, estimator_list, percentile_list]
        list_names = ['Scores', 'Classifiers', 'Number_of_Features', 'Way_of_Selection', 'Estimator', 'Percentile']
        
        percentile = self.params['percentile']
        clfs = self.params['clfs']
        feature_selection_type = self.params['feature_selection_type']
        
        if clfs == None :
            print(f'Performing nested cross-validation for {self.estimator.__class__.__name__}...')
            nested_scores_tr = self.inner_loop(X_train=X_train_selected, X_test=X_test_selected, y_train=y_train, y_test=y_test)
        else:
            for estimator in clfs:
                self.estimator = self.available_clfs[estimator]
                self.name = self.estimator.__class__.__name__
                print(f'Performing nested cross-validation for {self.estimator.__class__.__name__}...')
                nested_scores_tr = self.inner_loop(X_train=X_train_selected, X_test=X_test_selected, y_train=y_train, y_test=y_test)
                estimator_list.append(self.name)
                nested_scores.append(nested_scores_tr[0])

                if num_feature == 'full' or percentile == 100:
                    number_of_features.append(X_test_selected.shape[1])
                    way_of_selection.append('full')
                    classifiers_list.append(str(self.name))
                else:
                    way_of_selection.append(feature_selection_type)
                    if num_feature == None:
                        percentile_list.append(percentile)
                        classifiers_list.append(str(self.name)+'_'+feature_selection_type+'_'+str(percentile))
                    else:
                        number_of_features.append(num_feature)
                        classifiers_list.append(str(self.name)+'_'+feature_selection_type+'_'+str(num_feature))
        df_data = pd.DataFrame()
        for list_data, name in zip(lists, list_names):
            if len(list_data) != 0:  
                df_data[name] = list_data
        return df_data

    def _parallel_nested_cv_trial(self,i):
        inner_splits = self.params['inner_splits']
        outer_splits = self.params['outer_splits']
        num_features = self.params['num_features']
        percentile = self.params['percentile']
        feature_selection_type = self.params['feature_selection_type']
        feature_selection_method = self.params['feature_selection_method']
        list_dfs=[]
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)
        self.params['inner_cv'] = inner_cv
        self.params['outer_cv'] = outer_cv
            # outer_scorer = get_scorer(outer_scoring)
        j=1
        for train_index, test_index in outer_cv.split(self.X, self.y):
            # print(f'Finished {fini/(num_trials*outer_splits)*100}%')
            # fini+=1
            print(f'Outer fold {j} out of {outer_splits}')
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            if num_features is not None and percentile is None:
                if type(num_features) is int:
                    if num_features < X_train.shape[1]:
                        self.selected_features=self.feature_selection(X=X_train, y=y_train,method=feature_selection_type, n_features=num_features, inner_method=feature_selection_method)                            
                        X_train_selected = X_train[self.selected_features]
                        X_test_selected = X_test[self.selected_features]
                        num_feature=num_features
                    elif type(num_features) is int and num_features == X_train.shape[1]:
                        X_train_selected = X_train
                        X_test_selected = X_test
                        num_feature='full'
                    else: 
                        raise ValueError('num_features must be an integer less than the number of features in the dataset')
                    df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature)                        
                    list_dfs.append(df)
                elif type(num_features) is list and percentile is None:
                    for num_feature in num_features:
                        print(f'For {num_feature} features:')
                        if num_feature > X_train.shape[1]:
                            raise ValueError('num_features must be less than the number of features in the dataset')                            
                        elif num_feature == X_train.shape[1]:
                            X_train_selected = X_train
                            X_test_selected = X_test
                            num_feature='full'
                        else: 
                            self.selected_features=self.feature_selection(X=X_train, y=y_train, method=feature_selection_type, n_features=num_feature, inner_method=feature_selection_method)
                            X_train_selected = X_train[self.selected_features]
                            X_test_selected = X_test[self.selected_features]
                            num_feature=num_feature
                        df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature)
                        list_dfs.append(df)
            elif percentile is not None and num_features is None:
                if type(num_features) is int:
                    if percentile ==100:
                        X_train_selected = X_train
                        X_test_selected = X_test
                        num_feature='full'
                    elif percentile < 100 and percentile > 0:
                        self.selected_features=self.feature_selection(X=X_train, y=y_train,method=feature_selection_type, inner_method=feature_selection_method, percentile=percentile)
                        X_train_selected = X_train[self.selected_features]
                        X_test_selected = X_test[self.selected_features]
                        num_feature=percentile
                    else: 
                        raise ValueError('num_features must be an integer less or equal than 100 and hugher thatn 0')
                    df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature)
                    list_dfs.append(df)
            elif type(percentile) is list and num_features is None:
                for perc in percentile:
                    print(f'For percentile {perc}:')
                    if perc == 100: 
                        X_train_selected = X_train
                        X_test_selected = X_test
                        num_feature='full'
                    else:
                        self.selected_features=self.feature_selection(X=X_train, y=y_train,method=feature_selection_type, inner_method=feature_selection_method, percentile=percentile)
                        X_train_selected = X_train[self.selected_features]
                        X_test_selected = X_test[self.selected_features]
                        num_feature=num_feature
                    df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature)
                    list_dfs.append(df)
            elif percentile is None and num_features is None:
                X_train_selected = X_train
                X_test_selected = X_test
                num_feature='full'
                df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature)
                list_dfs.append(df)
            elif percentile is not None and num_features is not None:
                raise ValueError('Choose between num_of_features and percentile, one of them must be None')                    
            else:         
                raise ValueError('num_features must be an integer or a list or None')
            j+=1
        return list_dfs


    def nested_cross_validation(self, params):
        inner_scoring=params['inner_scoring']
        outer_scoring=params['outer_scoring']
        num_trials=params['num_trials']
        if inner_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        if outer_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        trial_indices = range(num_trials)  

        with Pool(processes=num_trials) as pool:
            list_dfs = list(tqdm(pool.imap(self._parallel_nested_cv_trial, trial_indices), total=num_trials))
        
        list_dfs_flat = list(chain.from_iterable(list_dfs))

    # Now you can concatenate them into a single DataFrame
        df_final = pd.DataFrame()
        for dataframe in list_dfs_flat:
            df_final = pd.concat([df_final, dataframe], axis=0)
    
        return df_final
    
    
    def model_selection(self,optimizer = 'grid_search', n_trials_ncv=25, n_trials=100, n_iter=25, 
                        num_trials=10, score = 'matthews_corrcoef', exclude=None,
                        search_on=None, return_scores_df=False, alpha=0.2,num_features=None,
                        feature_selection_type='mrmr',feature_selection_method='chi2', plot='box',percentile=None,
                        return_best_model=True,choose_model=False,inner_scoring='matthews_corrcoef',
                        outer_scoring='matthews_corrcoef',inner_splits=3, outer_splits=5, 
                        n_jobs=-1, verbose=0):
        """
        Perform model selection using Nested Cross Validation.

        Parameters:
            optimizer (str, optional): The optimization method to use. Defaults to 'grid_search'.
                Options: 'grid_search', 'random_search', 'bayesian_search'.
            n_trials_ncv (int, optional): The number of nested cross-validation splits. Defaults to 25.
            n_trials (int, optional): The number of trials for Bayesian optimization. Defaults to 100.
            n_iter (int, optional): The number of iterations for Random search. Defaults to 25.
            num_trials (int, optional): The number of cross-validation splits. Defaults to 10.
            score (str, optional): The scoring metric. Defaults to 'matthews_corrcoef'.
            exclude (list, optional): List of classifiers to exclude from selection. Defaults to None.
            search_on (list, optional): List of classifiers to include in selection. Defaults to None.
            scores_df (bool, optional): Whether to return a DataFrame of scores. Defaults to False.
            alpha (float, optional): The weight of standard deviation in the evaluation score. Defaults to 1.
            plot (str, optional): Type of plot for visualization ('box', 'violin', or None). Defaults to 'box'.
                Options: 'bayesian_search', 'grid_search', 'random_search', None. Defaults to None.
            return_model (bool, optional): Whether to return the best fitted estimator. Defaults to True.

        Returns:
            Fitted best estimator if return_model is True. Optionally returns a DataFrame of scores if scores_df is True.
            The return type depends on the flags `return_model` and `scores_df`:
                - If both are True, returns a tuple (best_estimator, scores_dataframe).
                - If only return_model is True, returns best_estimator.
                - If only scores_df is True, returns scores_dataframe.
                - Otherwise, returns None.
        """
        self.params = locals()
        self.params.pop('self', None)

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
        self.params['clfs'] = clfs

        df = self.nested_cross_validation(self.params)
        for classif in np.unique(df['Classifiers']):
            print(f'Resulting {classif}...')
            indices = df[df['Classifiers'] == classif]
            filtered_scores = indices['Scores'].values
            # filtered_best_params = indices['Best_Params'].values
    
            mean_score = np.mean(filtered_scores)
            max_score = np.max(filtered_scores)
            std_score = np.std(filtered_scores)
            median_score = np.median(filtered_scores)
            evaluated_score = mean_score - alpha * std_score  
            Numbers_of_Features = indices['Number_of_Features'].unique()[0]
            Way_of_Selection = indices['Way_of_Selection'].unique()[0]

            results.append({
                    'Estimator': df[df['Classifiers'] == classif]['Estimator'].unique()[0],
                    'Classifier': classif,
                    'Scores': filtered_scores.tolist(),  
                    'Max': max_score,
                    'Std': std_score,
                    'Median': median_score,
                    'Evaluated': evaluated_score,
                    # 'Best_Params': filtered_best_params.tolist(),
                    'Numbers_of_Features': Numbers_of_Features,
                    'Way_of_Selection': Way_of_Selection 
                })
        
        scores_dataframe = pd.DataFrame(results) if return_scores_df else None

        labels = scores_dataframe['Classifier'].unique() 
        x_coords = range(1, len(labels) + 1)
        evaluated_means = [scores_dataframe[scores_dataframe['Classifier'] == label]['Evaluated'] for label in labels]
        if plot == 'box':
            plt.boxplot(scores_dataframe['Scores'], labels=labels)
            x_coords = range(1, len(labels) + 1)
            plt.scatter( x_coords, evaluated_means, marker='*', color='red')
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(rotation=90)  
            plt.grid(True)
            plt.show()
        elif plot == 'violin':            
            plt.violinplot(scores_dataframe['Scores'].values)
            plt.scatter(x_coords,evaluated_means, marker='*', color='red')
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(x_coords, labels, rotation=90)
            plt.grid(True)
            plt.show()
        elif plot == None: pass
        else: raise ValueError(f'The "{plot}" is not a valid option for plotting. Choose between "box", "violin" or None.')
        
        if choose_model:
            unique_classifiers = scores_dataframe['Estimator'].unique()
            print("Select a classifier:")
            for idx, classifier in enumerate(unique_classifiers, start=1):
                print(f"{idx}: {classifier}")

            selection = int(input("Enter the number for your classifier choice: ")) - 1
            selected_classifier = unique_classifiers[selection]
            self.name = selected_classifier
            self.estimator = self.available_clfs[self.name] 

            features_options = list(scores_dataframe[scores_dataframe['Estimator'] == selected_classifier]['Numbers_of_Features'].unique())
            if self.X.shape[1] not in features_options:
                features_options.append(self.X.shape[1])
    
            print("Available number of features options:")
            for idx, features in enumerate(features_options, start=1):
                print(f"{idx}: {features}")

            feat_sel = int(input("Enter the number for your preferred number of features: ")) - 1
            selected_features = features_options[feat_sel]
            print(f"Selected classifier: {selected_classifier}, with {selected_features} features.")
            final_features = selected_features
        else:
            self.name = max(results, key=lambda x: x['Evaluated'])['Estimator']
            final_features = max(results, key=lambda x: x['Evaluated'])['Numbers_of_Features']
            self.estimator = self.available_clfs[self.name] 

        X2fit = self.X.copy()
        if final_features != X2fit.shape[1]:
            selected_X = self.feature_selection(X=X2fit,y=self.y, method = feature_selection_type, n_features = final_features, inner_method=feature_selection_method, percentile=percentile)
            X2fit = X2fit[selected_X]             
        else:
            pass
        
        if return_best_model:
            if optimizer == 'bayesian_search':
                self.best_estimator = self.bayesian_search(X=X2fit, y=self.y, cv=5, n_trials=n_trials, verbose=True,return_model=return_best_model)
            elif optimizer == 'grid_search':
                self.best_estimator = self.grid_search(X=X2fit, y=self.y, cv=5, verbose=True,return_model=return_best_model)
            elif optimizer == 'random_search':
                self.best_estimator = self.random_search(X=X2fit, y=self.y, cv=5, n_iter=n_iter, verbose=True,return_model=return_best_model)
            self.best_estimator = self.estimator.fit(X2fit, self.y)
            if return_scores_df:
                return self.best_estimator, scores_dataframe
            else: return self.best_estimator
        elif return_scores_df:
            return scores_dataframe
        else:
            print(f'Best estimator: {self.best_estimator}')
