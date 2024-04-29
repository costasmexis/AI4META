import logging
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import progressbar
import sklearn
from catboost import CatBoostClassifier
from joblib import Parallel, delayed, parallel_backend
from lightgbm import LGBMClassifier
from mrmr import mrmr_classif
from scipy.stats import sem
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.gaussian_process import GaussianProcessClassifier
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
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from xgboost import XGBClassifier

from dataloader import DataLoader

from .mlestimator import MachineLearningEstimator
from .optuna_grid import optuna_grid


class MLPipelines(MachineLearningEstimator):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        super().__init__(label, csv_dir, estimator, param_grid)
    
    def inner_loop(self, train_index, test_index, X, y, avail_thr):
        num_features = self.params['num_features']
        if isinstance(num_features, int):
            feature_loop = [num_features]
        elif isinstance(num_features, list):
            feature_loop = num_features
        elif num_features is None:
            feature_loop = [X.shape[1]]
        else: 
            raise ValueError('num_features must be an integer or a list or None')
        
        clfs = self.params['clfs']
        feature_selection_type = self.params['feature_selection_type']
        optimizer = self.params['optimizer']
        inner_scoring = self.params['inner_scoring']
        inner_cv = self.params['inner_cv']
        n_iter = self.params['n_iter']
        n_trials_ncv = self.params['n_trials_ncv']
        outer_scoring = self.params['outer_scoring']
        outer_scorer = get_scorer(outer_scoring)
        parallel = self.params['parallel']
        opt_grid = 'NestedCV_single'
        if parallel == 'thread_per_round':
            n_jobs = 1
        elif parallel == 'freely_parallel' or parallel == 'dynamic_parallel' :
            n_jobs = avail_thr
            
        results = {
            'Scores': [],
            'Classifiers': [],
            'Selected_Features': [],
            'Number_of_Features': [],
            'Hyperparameters': [],
            'Way_of_Selection': [],
            'Estimator': []
        }
        
        for num_feature2_use in feature_loop:
            X_train_selected, X_test_selected, num_feature = self.filter_features(train_index, test_index, X, y, num_feature2_use) 
            y_train, y_test = y[train_index], y[test_index]
            
            if clfs is None:
                raise ValueError("No classifier specified.")
            
            for estimator in clfs:
                self.estimator = self.available_clfs[estimator]
                self.name = self.estimator.__class__.__name__
                nested_scores = []

                if optimizer == 'grid_search':
                    clf = GridSearchCV(estimator=self.estimator, scoring=inner_scoring, 
                                       param_grid=self.param_grid, cv=inner_cv, n_jobs=1, verbose=0)
                elif optimizer == 'random_search':
                    clf = RandomizedSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                             param_distributions=self.param_grid, cv=inner_cv, n_jobs=1, 
                                             verbose=0, n_iter=n_iter)
                elif optimizer == 'bayesian_search':
                    self.set_optuna_verbosity(logging.ERROR)
                    clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                                            param_distributions=optuna_grid[opt_grid][self.name],
                                                            cv=inner_cv, n_jobs=n_jobs, verbose=0, n_trials=n_trials_ncv)
                else:
                    raise Exception("Unsupported optimizer.")   
                
                clf.fit(X_train_selected, y_train)

                nested_score = outer_scorer(clf, X_test_selected, y_test)
                nested_scores.append(nested_score)    
                
                results['Scores'].append(nested_scores[0])
                results['Estimator'].append(self.name)    
                results['Hyperparameters'].append(clf.best_params_)        
                
                if num_feature == 'full' or num_feature is None:
                    results['Selected_Features'].append(None)  
                    results['Number_of_Features'].append(X_test_selected.shape[1])
                    results['Way_of_Selection'].append('full')
                    results['Classifiers'].append(f"{self.name}")
                else:
                    results['Classifiers'].append(f"{self.name}_{feature_selection_type}_{num_feature}")
                    results['Selected_Features'].append(X_train_selected.columns.tolist())  
                    results['Number_of_Features'].append(num_feature)
                    results['Way_of_Selection'].append(feature_selection_type)

        time.sleep(1)
        return [results]
    
    def filter_features(self, train_index, test_index, X, y, num_feature2_use):
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        norm_method = self.params['norm_method']
        missing_values_method = self.params['missing_values_method']
        X_train, X_test = self.normalize(X=X_tr, train_test_set=True, X_test=X_te, method=norm_method)
        
        X_train = self.missing_values(data=X_train, method=missing_values_method, verbose=False)
        X_test = self.missing_values(data=X_test, method=missing_values_method, verbose=False)
        
        y_train, y_test = y[train_index], y[test_index]
        feature_selection_type = self.params['feature_selection_type']
        feature_selection_method = self.params['feature_selection_method']
        
        if num_feature2_use is not None and feature_selection_type != 'percentile':
            if isinstance(num_feature2_use, int):
                if num_feature2_use < X_train.shape[1]:
                    self.selected_features = self.feature_selection(X=X_train, y=y_train, method=feature_selection_type, num_features=num_feature2_use, inner_method=feature_selection_method)                            
                    X_train_selected = X_train[self.selected_features]
                    X_test_selected = X_test[self.selected_features]
                    num_feature = num_feature2_use
                elif num_feature2_use == X_train.shape[1]:
                    X_train_selected = X_train
                    X_test_selected = X_test
                    num_feature = 'full'
                else: 
                    raise ValueError('num_features must be an integer less than the number of features in the dataset')
        elif feature_selection_type == 'percentile' and num_feature2_use is not None:
            if isinstance(num_feature2_use, int):
                if num_feature2_use == 100:
                    X_train_selected = X_train
                    X_test_selected = X_test
                    num_feature = 'full'
                elif 0 < num_feature2_use < 100:
                    self.selected_features = self.feature_selection(X=X_train, y=y_train, method=feature_selection_type, inner_method=feature_selection_method, num_features=num_feature2_use)
                    X_train_selected = X_train[self.selected_features]
                    X_test_selected = X_test[self.selected_features]
                    num_feature = num_feature2_use
                else: 
                    raise ValueError('num_features must be an integer less than or equal to 100 and greater than 0')
        else:         
            raise ValueError('num_features must be an integer or None')
        
        return X_train_selected, X_test_selected, num_feature

    def outer_cv_loop(self,i,avail_thr):
            start = time.time()
            inner_splits = self.params['inner_splits']
            outer_splits = self.params['outer_splits']
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)
            self.params['inner_cv'] = inner_cv
            self.params['outer_cv'] = outer_cv 
            parallel = self.params['parallel']
            train_test_indices = list(outer_cv.split(self.X, self.y))
            list_dfs=[]
            widgets = [progressbar.Percentage()," ", progressbar.GranularBar(), " " ,
                        progressbar.Timer(), " ", progressbar.ETA()]
            
            if parallel == 'freely_parallel':
                temp_list = []
                with progressbar.ProgressBar(prefix = f'Outer fold of {i+1} round:',max_value=outer_splits, widgets=widgets) as bar:
                    split_index = 0
                    for train_index, test_index in train_test_indices:
                        results = self.inner_loop(train_index, test_index, self.X, self.y, avail_thr)
                        temp_list.append(results)
                        bar.update(split_index)
                        split_index += 1
                        time.sleep(1)
                    list_dfs = [item for sublist in temp_list for item in sublist]
                    end = time.time()
                    
            elif parallel == 'dynamic_parallel':
                temp_list = []
                results = Parallel(n_jobs=len(train_test_indices))(delayed(self.inner_loop)(
                    train_index, test_index, self.X, self.y, avail_thr)
                    for train_index, test_index in tqdm(train_test_indices,desc='Outer fold of %i round:'%(i+1),total=len(train_test_indices)))
                temp_list = [item for sublist in results for item in sublist]
                list_dfs.extend(temp_list) 
                end = time.time()
                
                
            else:
                temp_list = []
                with progressbar.ProgressBar(prefix = f'Outer fold of {i+1} round:',max_value=outer_splits, widgets=widgets) as bar:
                    split_index = 0
                    for train_index, test_index in train_test_indices:
                        results = self.inner_loop(train_index, test_index, self.X, self.y, avail_thr)
                        temp_list.append(results)
                        bar.update(split_index)
                        split_index += 1
                        time.sleep(1)
                    list_dfs = [item for sublist in temp_list for item in sublist]
                    end = time.time()
                
            print(f'Finished with {i+1} round after {(end-start)/3600:.2f} hours.')
            return list_dfs
    
    def nested_cv(self, optimizer='bayesian_search', n_trials_ncv=25, n_iter=25, 
                    rounds=10, exclude=None, hist_feat=True, N=100, most_imp_feat=10, include=None,
                    num_features=None, feature_selection_type='mrmr', return_csv=True,
                    feature_selection_method='chi2', plot='box', inner_scoring='matthews_corrcoef',
                    outer_scoring='matthews_corrcoef', inner_splits=5, outer_splits=5, norm_method='minmax',
                    parallel='thread_per_round', missing_values_method='median', return_all_N_features=True):
        """Performs nested cross-validation for machine learning pipelines.

        :param optimizer: The optimizer to use for hyperparameter tuning. Defaults to 'bayesian_search'.
        :type optimizer: str, optional
        :param n_trials_ncv: The number of trials for nested cross-validation. Defaults to 25.
        :type n_trials_ncv: int, optional
        :param n_iter: The number of iterations for hyperparameter tuning. Defaults to 25.
        :type n_iter: int, optional
        :param rounds: The number of rounds for nested cross-validation. Defaults to 10.
        :type rounds: int, optional
        :param exclude: A list of estimators to exclude from the cross-validation. Defaults to None.
        :type exclude: list, optional
        :param hist_feat: Whether to display histogram of selected features. Defaults to True.
        :type hist_feat: bool, optional
        :param N: The maximum number of features to display in the histogram. Defaults to 100.
        :type N: int, optional
        :param most_imp_feat: The number of most important features to display in the histogram. Defaults to 10.
        :type most_imp_feat: int, optional
        :param include: A list of estimators to include in the cross-validation. Defaults to None.
        :type include: list, optional
        :param num_features: The number of features to select. Defaults to None.
        :type num_features: int, optional
        :param feature_selection_type: The type of feature selection method to use. Defaults to 'mrmr'.
        :type feature_selection_type: str, optional
        :param return_csv: Whether to return the results as a CSV file. Defaults to True.
        :type return_csv: bool, optional
        :param feature_selection_method: The feature selection method to use. Defaults to 'chi2'.
        :type feature_selection_method: str, optional
        :param plot: The type of plot to display. Defaults to 'box'.
        :type plot: str, optional
        :param inner_scoring: The scoring metric to use for inner cross-validation. Defaults to 'matthews_corrcoef'.
        :type inner_scoring: str, optional
        :param outer_scoring: The scoring metric to use for outer cross-validation. Defaults to 'matthews_corrcoef'.
        :type outer_scoring: str, optional
        :param inner_splits: The number of splits for inner cross-validation. Defaults to 5.
        :type inner_splits: int, optional
        :param outer_splits: The number of splits for outer cross-validation. Defaults to 5.
        :type outer_splits: int, optional
        :param norm_method: The normalization method to use. Defaults to 'minmax'.
        :type norm_method: str, optional
        :param parallel: The parallelization method to use. Defaults to 'thread_per_round'.
        :type parallel: str, optional
        :param missing_values_method: The method to handle missing values. Defaults to 'median'.
        :type missing_values_method: str, optional
        :param return_all_N_features: Whether to return all N features. Defaults to True.
        :type return_all_N_features: bool, optional
        :return: The results of the nested cross-validation.
        :rtype: list
        """
        if missing_values_method == 'drop':
            print('Values cannot be dropped at ncv because of inconsistent shapes. \
                  The "median" will be used to handle missing values.')
            missing_values_method = 'median'

        self.params = locals()
        self.params.pop('self', None)

        all_scores = []
        results = []

        if exclude is not None:
            exclude_classes = [classifier.__class__ for classifier in exclude]
        elif include is not None:
            classes = [classifier.__class__ for classifier in include]
            exclude_classes = [classifier.__class__ for classifier in self.available_clfs.values() if
                               classifier.__class__ not in classes]
        else:
            exclude_classes = []

        clfs = [clf for clf in self.available_clfs.keys() if self.available_clfs[clf].__class__ not in exclude_classes]
        self.params['clfs'] = clfs

        if self.X.isnull().values.any():
            print('Your Dataset contains NaN values. Some estimators may not work with NaN values.')

        if inner_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        if outer_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')

        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        use_cores = min(num_cores, rounds)

        if parallel == 'thread_per_round':
            with threadpool_limits(limits=1):
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(self.outer_cv_loop)(i, 1) for i in trial_indices)

        elif parallel == 'dynamic_parallel':
            with Pool() as pool:
                list_dfs = pool.starmap(self.outer_cv_loop, [(i, 1) for i in trial_indices])

        elif parallel == 'freely_parallel':
            with threadpool_limits(limits=1):
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(self.outer_cv_loop)(i, 1) for i in trial_indices)

        else:
            raise ValueError(
                f'Invalid parallel option: {parallel}. Select one of the following: thread_per_round or freely_parallel')

        list_dfs_flat = list(chain.from_iterable(list_dfs))

        df = pd.concat([pd.DataFrame(item) for item in list_dfs_flat], axis=0)

        for classif in np.unique(df['Classifiers']):
            indices = df[df['Classifiers'] == classif]
            filtered_scores = indices['Scores'].values
            if num_features is not None:
                filtered_features = indices['Selected_Features'].values
            mean_score = np.mean(filtered_scores)
            max_score = np.max(filtered_scores)
            std_score = np.std(filtered_scores)
            sem_score = sem(filtered_scores)
            median_score = np.median(filtered_scores)
            Numbers_of_Features = indices['Number_of_Features'].unique()[0]
            Way_of_Selection = indices['Way_of_Selection'].unique()[0]
            results.append({
                'Estimator': df[df['Classifiers'] == classif]['Estimator'].unique()[0],
                'Classifier': classif,
                'Scores': filtered_scores.tolist(),
                'Max': max_score,
                'Std': std_score,
                'SEM': sem_score,
                'Median': median_score,
                'Hyperparameters': df[df['Classifiers'] == classif]['Hyperparameters'].values,
                'Selected_Features': filtered_features if num_features is not None else None,
                'Numbers_of_Features': Numbers_of_Features,
                'Way_of_Selection': Way_of_Selection
            })

        print(f'Finished with {len(results)} estimators')
        scores_dataframe = pd.DataFrame(results)

        feature_counts = Counter()
        for idx, row in scores_dataframe.iterrows():
            if row['Way_of_Selection'] != 'full':
                features = list(chain.from_iterable([list(index_obj) for index_obj in row['Selected_Features']]))
                feature_counts.update(features)

        sorted_features_counts = feature_counts.most_common()
        N = min(N, len(sorted_features_counts)) if N is not None else len(sorted_features_counts)

        if hist_feat:
            if len(sorted_features_counts) == 0:
                print('No features were selected.')
            else:
                features, counts = zip(*sorted_features_counts[:N])
                counts = [x / 12 for x in counts]
                plt.figure(figsize=(max(10, N // 2), 10))
                bars = plt.bar(range(N), counts, color='skyblue', tick_label=features)
                most_imp_feat = min(most_imp_feat, N) if most_imp_feat is not None else N
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
                s = maxsize / plt.gcf().dpi * N + 2 * m
                margin = m / plt.gcf().get_size_inches()[0]

                plt.gcf().subplots_adjust(left=margin, right=1. - margin)
                plt.gca().set_xticks(plt.gca().get_xticks()[::1])
                plt.gca().set_xlim([-1, N])

                plt.tight_layout()
                plt.show()

        all_N_features_list = [x[0] for x in sorted_features_counts]
        features_list = [x[0] for x in sorted_features_counts[:most_imp_feat]]

        if plot == 'box':
            plt.boxplot(scores_dataframe['Scores'], labels=scores_dataframe['Classifier'].unique())
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.show()
        elif plot == 'violin':
            plt.violinplot(scores_dataframe['Scores'].values)
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(range(1, len(scores_dataframe['Classifier'].unique()) + 1),
                       scores_dataframe['Classifier'].unique(), rotation=90)
            plt.grid(True)
            plt.show()

        if return_csv:
            scores_dataframe.to_csv('ncv_results.csv', index=False)
        if return_all_N_features:
            return scores_dataframe, features_list, all_N_features_list
        else:
            return scores_dataframe, features_list
