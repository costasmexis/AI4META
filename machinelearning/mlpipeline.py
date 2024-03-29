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
# from tqdm import tqdm
# from IPython.display import display
from tqdm import tqdm_notebook as tqdm
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
# from multiprocessing import Pool
from itertools import chain
import time
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed,  parallel_backend
# from progress.bar import Bar
import multiprocessing 
from collections import Counter
# from logging_levels import add_log_level
import progressbar
from progress.bar import Bar
from dask.distributed import Client, as_completed
import dask

    
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
        # self.set_optuna_verbosity(logging.ERROR)

    # def set_optuna_verbosity(self, level):
    #     """Adjust Optuna's verbosity level."""
    #     optuna.logging.set_verbosity(level)  
    #     logging.getLogger("optuna").setLevel(level) 
    # def no_display(*args, **kwargs):
    #     pass

    # @jit(parallel=True)
    def inner_loop(self, avail_thr, X_train=None, X_test=None, y_train=None, y_test=None):
        nested_scores = []
        self.set_optuna_verbosity(logging.ERROR)
        optimizer = self.params['optimizer']
        inner_scoring = self.params['inner_scoring']
        inner_cv = self.params['inner_cv']
        # n_jobs = self.params['n_jobs']
        # verbose = self.params['verbose']
        n_iter = self.params['n_iter']
        n_trials_ncv = self.params['n_trials_ncv']
        outer_scoring = self.params['outer_scoring']
        outer_scorer = get_scorer(outer_scoring)
        parallel = self.params['parallel']
        opt_grid = 'NestedCV_single'
        if parallel == 'thread_per_round':
            n_jobs = 1
            # opt_grid = 'NestedCV_single'
        elif parallel == 'freely_parallel' or parallel == 'dynamic_parallel':
            n_jobs = avail_thr
            # if parallel == 'dynamic_parallel':
            #     opt_grid = 'NestedCV_multi'
            # else:
           

        if optimizer == 'grid_search':
            clf = GridSearchCV(estimator=self.estimator, scoring=inner_scoring, 
                              param_grid=self.param_grid, cv=inner_cv, n_jobs=1, verbose=0)
        elif optimizer == 'random_search':
                    clf = RandomizedSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                             param_distributions=self.param_grid, cv=inner_cv, n_jobs=1, 
                                             verbose=0, n_iter=n_iter)
        elif optimizer == 'bayesian_search':
            clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                                    param_distributions=optuna_grid[opt_grid][self.name],
                                                    cv=inner_cv, n_jobs=n_jobs, verbose=0, n_trials=n_trials_ncv)
        else:
            raise Exception("Unsupported optimizer.")   
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        nested_score = outer_scorer(clf, X_test, y_test)
        nested_scores.append(nested_score)
        return nested_scores 
    
    
    # @jit(parallel=True)
    def clf_app(self, X_train_selected, X_test_selected, y_train, y_test, num_feature, avail_thr):
        nested_scores = []
        classifiers_list = []
        number_of_features = []
        way_of_selection = []
        estimator_list = []
        selected_features = []
        lists = [nested_scores, classifiers_list, selected_features, number_of_features, way_of_selection, estimator_list]
        list_names = ['Scores', 'Classifiers','Selected_Features', 'Number_of_Features', 'Way_of_Selection', 'Estimator']
        
        clfs = self.params['clfs']
        feature_selection_type = self.params['feature_selection_type']
        
        if clfs == None :
            nested_scores_tr = self.inner_loop(avail_thr=avail_thr, X_train=X_train_selected, X_test=X_test_selected, y_train=y_train, y_test=y_test)
        else:
            for estimator in clfs:
                self.estimator = self.available_clfs[estimator]
                self.name = self.estimator.__class__.__name__
                nested_scores_tr = self.inner_loop(avail_thr=avail_thr, X_train=X_train_selected, X_test=X_test_selected, y_train=y_train, y_test=y_test)
                estimator_list.append(self.name)
                nested_scores.append(nested_scores_tr[0])

                if num_feature == 'full' or None:
                    number_of_features.append(X_test_selected.shape[1])
                    way_of_selection.append('full')
                    classifiers_list.append(str(self.name))
                else:
                    way_of_selection.append(feature_selection_type)
                    selected_features.append(X_train_selected.columns)
                    number_of_features.append(num_feature)
                    classifiers_list.append(str(self.name)+'_'+feature_selection_type+'_'+str(num_feature))
        df_data = pd.DataFrame()
        for list_data, name in zip(lists, list_names):
            if len(list_data) != 0:  
                df_data[name] = list_data
        return df_data
    
    
    def _dynamic_parallel_nested_cv_trial(self,i,avail_thr):
        # with threadpool_limits(limits=avail_thr):
        #     list_dfs = self.outer_cv_loop(i,avail_thr)
        list_dfs = self.outer_cv_loop(i,avail_thr)
        return list_dfs
    
    def _freely_parallel_nested_cv_trial(self,i,avail_thr):
        with threadpool_limits(limits=avail_thr):
            list_dfs = self.outer_cv_loop(i,avail_thr+1)
        return list_dfs

    def _thread_per_round_nested_cv_trial(self,i):
        avail_thr = 1
        with threadpool_limits(limits=avail_thr):
            list_dfs = self.outer_cv_loop(i,avail_thr)
        return list_dfs

    def outer_cv_loop(self,i,avail_thr):
            start = time.time()
            inner_splits = self.params['inner_splits']
            outer_splits = self.params['outer_splits']
            num_features = self.params['num_features']
            feature_selection_type = self.params['feature_selection_type']
            feature_selection_method = self.params['feature_selection_method']
            list_dfs=[]
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)
            self.params['inner_cv'] = inner_cv
            self.params['outer_cv'] = outer_cv 
            widgets = [progressbar.Percentage()," ", progressbar.GranularBar(), " " ,
                       progressbar.Timer(), " ", progressbar.ETA()]
            with progressbar.ProgressBar(prefix = f'Outer fold of {i+1} round:', max_value=outer_splits,widgets=widgets) as bar:
            # with Bar('Processing...') as bar:
                split_index = 0
                # for train_index, test_index in tqdm(outer_cv.split(self.X, self.y), desc='Processing outer fold', total=outer_splits):
                for train_index, test_index in outer_cv.split(self.X, self.y):
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y[train_index], self.y[test_index]
                    if num_features is not None and feature_selection_type != 'percentile':
                        if type(num_features) is int:
                            if num_features < X_train.shape[1]:
                                self.selected_features=self.feature_selection(X=X_train, y=y_train,method=feature_selection_type, num_features=num_features, inner_method=feature_selection_method)                            
                                X_train_selected = X_train[self.selected_features]
                                X_test_selected = X_test[self.selected_features]
                                num_feature=num_features
                            elif type(num_features) is int and num_features == X_train.shape[1]:
                                X_train_selected = X_train
                                X_test_selected = X_test
                                num_feature='full'
                            else: 
                                raise ValueError('num_features must be an integer less than the number of features in the dataset')
                            df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature,avail_thr=avail_thr)                        
                            list_dfs.append(df)
                        elif type(num_features) is list :
                            for num_feature in num_features:
                                if num_feature > X_train.shape[1]:
                                    raise ValueError('num_features must be less than the number of features in the dataset')                            
                                elif num_feature == X_train.shape[1]:
                                    X_train_selected = X_train
                                    X_test_selected = X_test
                                    num_feature='full'
                                else: 
                                    self.selected_features=self.feature_selection(X=X_train, y=y_train, method=feature_selection_type, num_features=num_feature, inner_method=feature_selection_method)
                                    X_train_selected = X_train[self.selected_features]
                                    X_test_selected = X_test[self.selected_features]
                                    num_feature=num_feature
                                df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature,avail_thr=avail_thr)
                                list_dfs.append(df)
                    elif feature_selection_type == 'percentile' and num_features is not None:
                        if type(num_features) is int:
                            if num_features == 100:
                                X_train_selected = X_train
                                X_test_selected = X_test
                                num_feature='full'
                            elif num_features < 100 and num_features > 0:
                                self.selected_features=self.feature_selection(X=X_train, y=y_train,method=feature_selection_type, inner_method=feature_selection_method, num_features=num_features)
                                X_train_selected = X_train[self.selected_features]
                                X_test_selected = X_test[self.selected_features]
                                num_feature=num_features
                            else: 
                                raise ValueError('num_features must be an integer less or equal than 100 and hugher thatn 0')
                            df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature,avail_thr=avail_thr)
                            list_dfs.append(df)
                        elif type(num_features) is list:
                            for num_feature in num_features:
                                if num_feature == 100: 
                                    X_train_selected = X_train
                                    X_test_selected = X_test
                                    num_feature='full'
                                else:
                                    self.selected_features=self.feature_selection(X=X_train, y=y_train, method=feature_selection_type, inner_method=feature_selection_method, num_features=num_feature)
                                    X_train_selected = X_train[self.selected_features]
                                    X_test_selected = X_test[self.selected_features]
                                    num_feature=num_feature
                                df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature,avail_thr=avail_thr)
                                list_dfs.append(df)
                    elif num_features is None:
                        X_train_selected = X_train
                        X_test_selected = X_test
                        num_feature='full'
                        df = self.clf_app(X_train_selected=X_train_selected, y_train=y_train, X_test_selected=X_test_selected, y_test=y_test,num_feature=num_feature,avail_thr=avail_thr)
                        list_dfs.append(df)
                    else:         
                        raise ValueError('num_features must be an integer or a list or None')
                    bar.update(split_index)
                    split_index += 1
                    time.sleep(1)
            end = time.time()
            print(f'Finished with {i+1} round after {(end-start)/3600:.2f} hours.')
            return list_dfs


    def nested_cross_validation(self, params):
        inner_scoring=params['inner_scoring']
        outer_scoring=params['outer_scoring']
        rounds=params['rounds']
        parallel=params['parallel']
        if inner_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        if outer_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        trial_indices = range(rounds)

        rounds = self.params['rounds']
        num_cores = multiprocessing.cpu_count()
        if num_cores < rounds:
            use_cores = num_cores
        else:
            use_cores = rounds
            
        avail_thr = num_cores//rounds
        if avail_thr == 0:
            avail_thr = 1

        if parallel == 'thread_per_round':
            list_dfs = Parallel(n_jobs=use_cores,verbose=0)(delayed(self._thread_per_round_nested_cv_trial)(i) for i in trial_indices)
        elif parallel == 'dynamic_parallel':                
            client = Client()
            print(f'You can observe the progress of the computation on the dashboard: \n{client.dashboard_link}') 
            list_dfs=[]
            futures = [client.submit(self._dynamic_parallel_nested_cv_trial, i, avail_thr) for i in trial_indices]
            for future in as_completed(futures):
                result = future.result() 
                list_dfs.append(result)
            client.close()
        elif parallel == 'freely_parallel':
            list_dfs = Parallel(n_jobs=use_cores,verbose=0)(delayed(self._dynamic_parallel_nested_cv_trial)(i,avail_thr) for i in trial_indices)
        else: raise ValueError(f'Invalid parallel option: {parallel}. Select one of the following: thread_per_round or freely_parallel')
        list_dfs_flat = list(chain.from_iterable(list_dfs))
        df_final = pd.DataFrame()
        for dataframe in list_dfs_flat:
            df_final = pd.concat([df_final, dataframe], axis=0)
        return df_final
    
    
    def model_selection(self,optimizer = 'grid_search', n_trials_ncv=25, n_trials=100, n_iter=25, 
                        rounds=10, score = 'matthews_corrcoef', exclude=None,hist_fit=True,N=100,
                        most_imp_feat=10,search_on=None, return_scores_df=False, alpha=0.2,num_features=None,
                        feature_selection_type='mrmr',feature_selection_method='chi2', plot='box',
                        return_best_model=True,choose_model=False,inner_scoring='matthews_corrcoef',
                        outer_scoring='matthews_corrcoef',inner_splits=5, outer_splits=5, verbose=True, parallel='thread_per_round'):
        """
        Perform model selection using Nested Cross Validation and visualize the selected features' frequency.

        Parameters:
            optimizer (str, optional): Optimization method used ('grid_search', 'random_search', 'bayesian_search'). 
                                       Defaults to 'grid_search'.
            n_trials_ncv (int, optional): Number of trials for nested cross-validation. Defaults to 25.
            n_trials (int, optional): Number of trials for Bayesian optimization. Defaults to 100.
            n_iter (int, optional): Number of iterations for random search. Defaults to 25.
            rounds (int, optional): Number of cross-validation splits. Defaults to 10.
            score (str, optional): Scoring metric used. Defaults to 'matthews_corrcoef'.
            exclude (list, optional): List of classifiers to exclude. Defaults to None.
            hist_fit (bool, optional): Whether to display a histogram of feature selection frequency. Defaults to True.
            N (int, optional): Number of features to display in the histogram. Defaults to None (all features).
            most_imp_feat (int, optional): Number of most important features highlighted in the histogram. Defaults to 10.
            search_on (list, optional): List of classifiers to include in selection. Defaults to None.
            return_scores_df (bool, optional): Whether to return a DataFrame of scores. Defaults to False.
            alpha (float, optional): Weight of standard deviation in evaluation score calculation. Defaults to 0.2.
            num_features (int or list, optional): Number of features to consider. Defaults to None (all features).
            feature_selection_type (str, optional): Method of feature selection ('mrmr', etc.). Defaults to 'mrmr'.
            feature_selection_method (str, optional): Method used within feature selection (e.g., 'chi2'). Defaults to 'chi2'.
            plot (str, optional): Type of plot for visualization ('box', 'violin', None). Defaults to 'box'.
            return_best_model (bool, optional): Whether to return the best fitted model. Defaults to True.
            choose_model (bool, optional): Allows manual selection of the model from the console. Defaults to False.
            inner_scoring (str, optional): Scoring metric for inner CV. Defaults to 'matthews_corrcoef'.
            outer_scoring (str, optional): Scoring metric for outer CV. Defaults to 'matthews_corrcoef'.
            inner_splits (int, optional): Number of splits for inner CV. Defaults to 5.
            outer_splits (int, optional): Number of splits for outer CV. Defaults to 5.
            verbose (bool, optional): Enables detailed logging. Defaults to True.
            parallel (str, optional): Parallelization method ('thread_per_round', 'thread_per_fold', 'freely_parallel'). Defaults to 'thread_per_round'.

        Returns:
            The best fitted estimator if return_best_model is True. Optionally returns a DataFrame of scores if return_scores_df is True. The exact return type depends on the flags `return_best_model` and `return_scores_df`:
                - If both are True, returns a tuple (best_estimator, scores_dataframe).
                - If only return_best_model is True, returns best_estimator.
                - If only return_scores_df is True, returns scores_dataframe.
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

        if self.X.isnull().values.any():
            print('Your Dataset contains NaN values. Some estimators does not work with NaN values.')

        df = self.nested_cross_validation(self.params)
        
        for classif in np.unique(df['Classifiers']):
            indices = df[df['Classifiers'] == classif]
            filtered_scores = indices['Scores'].values
            if num_features is not None:
                filtered_features = indices['Selected_Features'].values
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
                    'Selected_Features': filtered_features if num_features is not None else None,
                    'Numbers_of_Features': Numbers_of_Features,
                    'Way_of_Selection': Way_of_Selection 
                })
            
        print(f'Finished with {len(results)} estimators')
        scores_dataframe = pd.DataFrame(results) if return_scores_df else None
        
        if hist_fit:
            feature_counts = Counter()
            for idx, row in scores_dataframe.iterrows():
                if row['Way_of_Selection'] != 'full':
                    features = list(chain.from_iterable([list(index_obj) for index_obj in row['Selected_Features']]))
                    feature_counts.update(features)
            
            sorted_features_counts = feature_counts.most_common()
            if N is None or N > len(sorted_features_counts):
                N = len(sorted_features_counts)  # Adjust N as needed to limit the number of features displayed
            else:
                N=N
            if len(sorted_features_counts) == 0:
                print('No features were selected.')
            else:
                features, counts = zip(*sorted_features_counts[:N])
                plt.figure(figsize=(max(10, N // 2), 10))
                bars = plt.bar(range(N), counts, color='skyblue', tick_label=features)
                if most_imp_feat > N:
                    most_imp_feat = N
                elif most_imp_feat > 0 and most_imp_feat <= N and most_imp_feat != None:
                    for bar in bars[:most_imp_feat]:
                        bar.set_color('red')

                plt.xlabel('Features')
                plt.ylabel('Counts')
                plt.title('Histogram of Selected Features')
                plt.xticks(rotation=90)  

                plt.gca().margins(x=0.05)
                plt.gcf().canvas.draw()
                tl = plt.gca().get_xticklabels()
                maxsize = max([t.get_window_extent().width for t in tl])
                m = 0.5
                s = maxsize/plt.gcf().dpi*N+2*m
                margin = m/plt.gcf().get_size_inches()[0]

                plt.gcf().subplots_adjust(left=margin, right=1.-margin)
                plt.gca().set_xticks(plt.gca().get_xticks()[::1]) 
                plt.gca().set_xlim([-1, N])

                plt.tight_layout()
                plt.show()

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
            selected_X = self.feature_selection(X=X2fit,y=self.y, method = feature_selection_type, num_features = final_features, inner_method=feature_selection_method)
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