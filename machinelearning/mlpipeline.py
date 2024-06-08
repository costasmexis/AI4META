import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import sklearn
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
import os
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from mrmr import mrmr_classif
import logging
from scipy.stats import sem
# from .Features_explanation import Features_explanation
from multiprocessing import Pool
from itertools import chain
import time
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed,  parallel_backend
import multiprocessing 
from collections import Counter
import progressbar
from concurrent.futures import ThreadPoolExecutor, as_completed


    
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
    
    def set_result_csv_name(self, csv_dir):
        data_name = os.path.basename(csv_dir).split('.')[0]
        return data_name

    def inner_loop(self, train_index, test_index, X, y, avail_thr):
        num_features = self.params['num_features']
        inner_selection = self.params['inner_selection']

        if type(num_features) is int:
            feature_loop = [num_features]
        elif type(num_features) is list:
            feature_loop = num_features
        elif num_features is None:
            feature_loop = [X.shape[1]]
        else: raise ValueError('num_features must be an integer or a list or None')
        
        clfs = self.params['clfs']
        feature_selection_type = self.params['feature_selection_type']
        
        inner_scoring = self.params['inner_scoring']
        inner_cv = self.params['inner_cv']
        n_trials_ncv = self.params['n_trials_ncv']
        outer_scoring = self.params['outer_scoring']
        outer_scorer = get_scorer(outer_scoring)
        parallel = self.params['parallel']

        opt_grid = 'NestedCV'
        if parallel == 'thread_per_round':
            n_jobs = 1
        elif parallel == 'freely_parallel' or parallel == 'dynamic_parallel' :
            n_jobs = avail_thr
        # if parallel == 'dynamic_parallel':
            # opt_grid = 'NestedCV_multi'
            # n_jobs = 1
        # opt_grid = 'NestedCV_single'
            
        results={'Scores': [],
            'Classifiers': [],
            'Selected_Features': [],
            'Number_of_Features': [],
            'Hyperparameters': [],
            'Way_of_Selection': [],
            'Estimator': []}
        
        for num_feature2_use in feature_loop:
            X_train_selected, X_test_selected, num_feature = self.filter_features(train_index, test_index, X, y, num_feature2_use) 
            y_train, y_test = y[train_index], y[test_index]
            
            if clfs == None :
                raise ValueError("No classifier specified.")
            else:
                for estimator in clfs:
                    self.estimator = self.available_clfs[estimator]
                    self.name = self.estimator.__class__.__name__
                    
                    self.set_optuna_verbosity(logging.ERROR)
                    clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
                                                            param_distributions=optuna_grid[opt_grid][self.name],
                                                            cv=inner_cv, n_jobs=n_jobs, verbose=0, n_trials=n_trials_ncv)
                                        
                    clf.fit(X_train_selected, y_train)
                    results['Estimator'].append(self.name)
                    
                    if inner_selection == 'validation_score':
                        results['Scores'].append(outer_scorer(clf, X_test_selected, y_test))
                        results['Hyperparameters'].append(clf.best_params_)  
                    else:  
                        trials = clf.trials_
                        simple_model_params = self.one_sem_model(trials, self.name)
                        results['Hyperparameters'].append(simple_model_params) 
                        new_params_clf = self.create_model_instance(self.name, simple_model_params)
                        new_params_clf.fit(X_train_selected, y_train)
                        results['Scores'].append(new_params_clf.score(X_test_selected, y_test))

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

        df_results = pd.DataFrame(results)
        time.sleep(1)
        return [results]
    
    def create_model_instance(self, model_name, params):
        if model_name == 'RandomForestClassifier':
            return RandomForestClassifier(**params)
        elif model_name == 'LogisticRegression':
            return LogisticRegression(**params)
        elif model_name == 'XGBClassifier':
            return XGBClassifier(**params)
        elif model_name == 'LGBMClassifier':
            return LGBMClassifier(**params)
        elif model_name == 'CatBoostClassifier':
            return CatBoostClassifier(**params)
        elif model_name == 'SVC':
            return SVC(**params)
        elif model_name == 'KNeighborsClassifier':
            return KNeighborsClassifier(**params)
        elif model_name == 'LinearDiscriminantAnalysis':
            return LinearDiscriminantAnalysis(**params)
        elif model_name == 'GaussianNB':
            return GaussianNB(**params)
        elif model_name == 'GradientBoostingClassifier':
            return GradientBoostingClassifier(**params)
        elif model_name == 'GaussianProcessClassifier':
            return GaussianProcessClassifier(**params)
        elif model_name == 'DecisionTreeClassifier':
            return DecisionTreeClassifier(**params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    
    def one_sem_model(self, trials, model_name):
        hyper_compl = {
            'RandomForestClassifier': {'n_estimators': True, 'max_depth': True, 'min_samples_split': False, 'min_samples_leaf': False},
            'LogisticRegression': {'C': True, 'max_iter': True},
            'XGBClassifier': {'max_depth': True, 'n_estimators': True, 'learning_rate': True, 'gamma': False, 'min_child_weight': False, 'colsample_bytree': True, 'subsample': True, 'reg_lambda': False},
            'LGBMClassifier': {'max_depth': True, 'n_estimators': True, 'learning_rate': True, 'num_leaves': True, 'colsample_bytree': True, 'subsample': True, 'reg_lambda': False},
            'CatBoostClassifier': {'max_depth': True, 'n_estimators': True, 'learning_rate': True,  'reg_lambda': False},
            'SVC': {'C': True, 'degree': True},
            'KNeighborsClassifier': {'n_neighbors': True},
            'LinearDiscriminantAnalysis': {'shrinkage': True},
            'GaussianNB': {'var_smoothing': False},
            'GradientBoostingClassifier': {'n_estimators': True, 'max_depth': True, 'min_samples_split': False, 'min_samples_leaf': False},
            'GaussianProcessClassifier': {'max_iter_predict': True},
            'DecisionTreeClassifier': {'max_depth': True, 'min_samples_split': False, 'min_samples_leaf': False}
        }
        
        constraints = hyper_compl[model_name]
        inner_cv_splits = self.params['inner_splits']  # Number of splits in the inner CV
        trials_data = [{'params': t.params, 'value': t.values[0], 'sem': t.user_attrs['std_test_score'] / (inner_cv_splits ** 0.5)} for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        # trials_data = [{'params': t.params, 'value': t.values[0], 'sem': t.user_attrs['std_test_score']} for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        trials_data = sorted(trials_data, key=lambda x: (x['value'], -x['sem']), reverse=True)
        
        best_score = trials_data[0]['value']
        best_score_sem = trials_data[0]['sem']
        sem_threshold = best_score - best_score_sem
        
        filtered_trials = [t for t in trials_data if t['value'] >= sem_threshold]
        
        def model_complexity(params):
            complexity = 0
            param_ranges = optuna_grid['param_ranges']
            for p in constraints:
                range_min, range_max = param_ranges.get(p, (0, 1))  # Default range (0, 1) if not specified
                normalized_value = (params.get(p, 0) - range_min) / (range_max - range_min)
                if constraints[p]:
                    complexity += normalized_value
                else:
                    complexity -= normalized_value
            return complexity

        simplest_model = min(filtered_trials, key=lambda x: model_complexity(x['params']))
        
        return simplest_model['params']
    
    def filter_features(self, train_index, test_index, X, y, num_feature2_use):
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        norm_method = self.params['norm_method']
        missing_values_method = self.params['missing_values_method']
        X_train, X_test = self.normalize(X=X_tr,train_test_set=True,X_test=X_te, method=norm_method)
        
        X_train = self.missing_values(data=X_train, method=missing_values_method)
        X_test = self.missing_values(data=X_test, method=missing_values_method)
        
        y_train, y_test = y[train_index], y[test_index]
        feature_selection_type = self.params['feature_selection_type']
        feature_selection_method = self.params['feature_selection_method']
        
        if num_feature2_use is not None and feature_selection_type != 'percentile':
            if type(num_feature2_use) is int:
                if num_feature2_use < X_train.shape[1]:
                    self.selected_features=self.feature_selection(X=X_train, y=y_train,method=feature_selection_type, num_features=num_feature2_use, inner_method=feature_selection_method)                            
                    X_train_selected = X_train[self.selected_features]
                    X_test_selected = X_test[self.selected_features]
                    num_feature = num_feature2_use
                elif type(num_feature2_use) is int and num_feature2_use == X_train.shape[1]:
                    X_train_selected = X_train
                    X_test_selected = X_test
                    num_feature='full'
                else: 
                    raise ValueError('num_features must be an integer less than the number of features in the dataset')
        elif feature_selection_type == 'percentile' and num_feature2_use is not None:
            if type(num_feature2_use) is int:
                if num_feature2_use == 100:
                    X_train_selected = X_train
                    X_test_selected = X_test
                    num_feature='full'
                elif num_feature2_use < 100 and num_feature2_use > 0:
                    self.selected_features=self.feature_selection(X=X_train, y=y_train,method=feature_selection_type, inner_method=feature_selection_method, num_features=num_feature2_use)
                    X_train_selected = X_train[self.selected_features]
                    X_test_selected = X_test[self.selected_features]
                    num_feature=num_feature2_use
                else: 
                    raise ValueError('num_features must be an integer less or equal than 100 and hugher thatn 0')
        else:         
            raise ValueError('num_features must be an integer or a list or None')
        
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
                    
            # elif parallel == 'dynamic_parallel':
            #     temp_list = []
            #     results = Parallel(n_jobs=len(train_test_indices))(delayed(self.inner_loop)(
            #         train_index, test_index, self.X, self.y, avail_thr)
            #         for train_index, test_index in tqdm(train_test_indices,desc='Outer fold of %i round:'%(i+1),total=len(train_test_indices)))
            #     temp_list = [item for sublist in results for item in sublist]
            #     list_dfs.extend(temp_list) 
            #     end = time.time()
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
    
    def nested_cv(self,n_trials_ncv=25,rounds=10, exclude=None,hist_feat=True,N=100,most_imp_feat=10,search_on=None,
                    num_features=None,feature_selection_type='mrmr', return_csv=True, hist_fit=False,
                    feature_selection_method='chi2', plot='box',inner_scoring='matthews_corrcoef',inner_selection='validation_score',
                    outer_scoring='matthews_corrcoef',inner_splits=5, outer_splits=5,norm_method='minmax',
                    parallel='thread_per_round', missing_values_method='median',return_all_N_features=True):
        """
        Perform model selection using Nested Cross Validation and visualize the selected features' frequency.

        Parameters:
            # optimizer (str, optional): Optimization method used ('grid_search', 'random_search', 'bayesian_search'). 
            #                            Defaults to 'grid_search'.
            n_trials_ncv (int, optional): Number of trials for nested cross-validation. Defaults to 25.
            # n_iter (int, optional): Number of iterations for random search. Defaults to 25.
            rounds (int, optional): Number of cross-validation splits. Defaults to 10.
            exclude (list, optional): List of classifiers to exclude. Defaults to None.
            hist_fit (bool, optional): Whether to display a histogram of feature selection frequency. Defaults to True.
            N (int, optional): Number of features to display in the histogram. Defaults to None (all features).
            most_imp_feat (int, optional): Number of most important features highlighted in the histogram. Defaults to 10.
            search_on (list, optional): List of classifiers to include in selection. Defaults to None.
            num_features (int or list, optional): Number of features to consider. Defaults to None (all features).
            feature_selection_type (str, optional): Method of feature selection ('mrmr', etc.). Defaults to 'mrmr'.
            feature_selection_method (str, optional): Method used within feature selection (e.g., 'chi2'). Defaults to 'chi2'.
            plot (str, optional): Type of plot for visualization ('box', 'violin', None). Defaults to 'box'.
            inner_scoring (str, optional): Scoring metric for inner CV. Defaults to 'matthews_corrcoef'.
            outer_scoring (str, optional): Scoring metric for outer CV. Defaults to 'matthews_corrcoef'.
            inner_splits (int, optional): Number of splits for inner CV. Defaults to 5.
            outer_splits (int, optional): Number of splits for outer CV. Defaults to 5.
            parallel (str, optional): Parallelization method ('thread_per_round', 'freely_parallel' and  'dynamic_parallel'). Defaults to 'thread_per_round'.

        Returns:
            The best fitted estimator if return_best_model is True. Optionally returns a DataFrame of scores if return_scores_df is True. The exact return type depends on the flags `return_best_model` and `return_scores_df`:
                - If both are True, returns a tuple (best_estimator, scores_dataframe).
                - If only return_best_model is True, returns best_estimator.
                - If only return_scores_df is True, returns scores_dataframe.
                - Otherwise, returns None.
        """
        if missing_values_method == 'drop':
            print(f'Values cannot be dropped at ncv because of inconsistent shapes. The "median" will be used to handle missing values.')
            missing_values_method = 'median'
            
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

        if inner_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        if outer_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(f'Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}')
        if inner_selection not in ['validation_score', 'one_sem']:
            raise ValueError(f'Invalid inner method: {inner_selection}. Select one of the following: ["validation_score", "one_sem"]')
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        if num_cores < rounds:
            use_cores = num_cores
        else:
            use_cores = rounds
            
        avail_thr = max(1, num_cores//rounds)

        if parallel == 'thread_per_round':
            avail_thr = 1
            with threadpool_limits(limits=avail_thr):
                list_dfs = Parallel(n_jobs=use_cores,verbose=0)(delayed(self.outer_cv_loop)(i,avail_thr) for i in trial_indices)
        
        # elif parallel == 'dynamic_parallel': 
        #     # avail_thr = 1
        #     with Pool() as pool:
        #         list_dfs = pool.starmap(self.outer_cv_loop, [(i, avail_thr) for i in trial_indices])
                
        elif parallel == 'freely_parallel':
            # with threadpool_limits(limits=avail_thr):
            with threadpool_limits():
                list_dfs = Parallel(n_jobs=use_cores,verbose=0)(delayed(self.outer_cv_loop)(i,avail_thr) for i in trial_indices)
        
        else: raise ValueError(f'Invalid parallel option: {parallel}. Select one of the following: thread_per_round or freely_parallel')

        list_dfs_flat = list(chain.from_iterable(list_dfs))
            
        df = pd.DataFrame()
        for item in list_dfs_flat:
            dataframe = pd.DataFrame(item)
            df = pd.concat([df, dataframe], axis=0)        
        
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
                    'SEM':sem_score,
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
        if N is None or N > len(sorted_features_counts):
            N = len(sorted_features_counts)  # Adjust N as needed to limit the number of features displayed
        else:
            N=N
            
        if hist_feat:
            if len(sorted_features_counts) == 0:
                print('No features were selected.')
            else:
                features, counts = zip(*sorted_features_counts[:N])
                counts = [x / len(clfs) for x in counts]
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

        all_N_features_list = [x[0] for x in sorted_features_counts]
        features_list = [x[0] for x in sorted_features_counts[:most_imp_feat]]
        
        def bootstrap_median_ci(data, num_iterations=1000, ci=0.95):
            medians = []
            for _ in range(num_iterations):
                sample = np.random.choice(data, size=len(data), replace=True)
                medians.append(np.median(sample))
            lower_bound = np.percentile(medians, (1-ci)/2 * 100)
            upper_bound = np.percentile(medians, (1+ci)/2 * 100)
            return lower_bound, upper_bound
        
        # creste a results directory
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if plot != None: 
                    
            scores_long = scores_dataframe.explode('Scores')
            scores_long['Scores'] = scores_long['Scores'].astype(float)
            
            fig = go.Figure()
            
            if plot == 'box':
            # Add box plots for each classifier
                for classifier in scores_dataframe['Classifier']:
                    data = scores_long[scores_long['Classifier'] == classifier]['Scores']
                    fig.add_trace(go.Box(
                        y=data,
                        name=classifier,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                    
                    # Calculate and add 95% CI for the median
                    lower, upper = bootstrap_median_ci(data)
                    fig.add_trace(go.Scatter(
                        x=[classifier, classifier],
                        y=[lower, upper],
                        mode='lines',
                        line=dict(color='black', dash='dash'),
                        showlegend=False
                    ))

            elif plot == 'violin':
                for classifier in scores_dataframe['Classifier']:
                    data = scores_long[scores_long['Classifier'] == classifier]['Scores']
                    fig.add_trace(go.Violin(
                        y=data,
                        name=classifier,
                        box_visible=False,
                        points='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                            
                
            else: raise ValueError(f'The "{plot}" is not a valid option for plotting. Choose between "box" or "violin".')

            # Update layout for better readability
            fig.update_layout(
                title="Model Selection Results",
                yaxis_title="Score",
                xaxis_title="Classifier",
                xaxis_tickangle=-45,
                template="plotly_white"
            )
            # Save the figure as an image in the "Results" directory
            image_path = os.path.join(results_dir, "model_selection_results.png")
            fig.write_image(image_path)

            fig.show()
            
        else: pass
        
        if return_csv:
            try:
                dataset_name = self.set_result_csv_name(self.csv_dir)
                results_path = os.path.join(results_dir, f'{dataset_name}_ncv_results.csv')
                scores_dataframe.to_csv(results_path, index=False)
            except Exception as e:
                dataset_name = 'results_ncv'
                results_path = os.path.join(results_dir, f'{dataset_name}.csv')
                scores_dataframe.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")
        if return_all_N_features:
            return scores_dataframe, features_list, all_N_features_list
        else:
            return scores_dataframe, features_list