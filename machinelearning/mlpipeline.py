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

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from mrmr import mrmr_classif
import logging
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
        self.set_optuna_verbosity(logging.ERROR)

    def set_optuna_verbosity(self, level):
        """Adjust Optuna's verbosity level."""
        optuna.logging.set_verbosity(level)  
        logging.getLogger("optuna").setLevel(level) 

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
    
    def inner_loop(self, optimizer, inner_scoring, inner_cv, n_jobs, verbose,
                    n_iter, n_trials,X_train, y_train, X_test, y_test,
                    best_params_list, nested_scores, outer_scorer):
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
                                                    verbose=verbose, n_trials=n_trials)
        else:
            raise Exception("Unsupported optimizer.")   
        
        # return clf
        clf.fit(X_train, y_train)
        best_params_list.append(clf.best_params_)
        y_pred = clf.predict(X_test)
        nested_score = outer_scorer(clf, X_test, y_test)
        nested_scores.append(nested_score)
        return nested_scores, best_params_list
    
    

    def nested_cross_validation(self, inner_scoring='matthews_corrcoef', outer_scoring='matthews_corrcoef',
                                inner_splits=3, outer_splits=5, optimizer='grid_search', 
                                n_trials=100, n_iter=25, num_trials=10, n_jobs=-1, verbose=0,
                                num_features=None, feature_selection_type='mrmr'):
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
        classifiers_list = []
        number_of_features = []
        way_of_selection = []

        for i in range(num_trials):
            print(f'Trial {i+1} out of {num_trials}')
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=i)
            outer_scorer = get_scorer(outer_scoring)
            j=1
            for train_index, test_index in outer_cv.split(self.X, self.y):
                print(f'Outer fold {j} out of {outer_splits}')
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                if num_features is not None:
                    if type(num_features) is int:
                        if num_features < X_train.shape[1]:
                            self.selected_features = mrmr_classif(X_train, y_train, K=num_features)
                            X_train_selected = X_train[self.selected_features]
                            X_test_selected = X_test[self.selected_features]
                            nested_scores_tr, best_params_list_tr = self.inner_loop(optimizer, inner_scoring, inner_cv, n_jobs, verbose,
                                                                           n_iter, n_trials,X_train_selected, y_train, X_test_selected, y_test,
                                                                           best_params_list, nested_scores, outer_scorer)
                            classifiers_list.append(str(self.name)+'_'+str(num_features)+'_'+str(feature_selection_type))
                            number_of_features.append(num_features)
                            way_of_selection.append(feature_selection_type)
                        elif type(num_features) is int and num_features == X_train.shape[1]:
                            nested_scores_tr, best_params_list_tr = self.inner_loop(optimizer, inner_scoring, inner_cv, n_jobs, verbose,
                                                                           n_iter, n_trials,X_train, y_train, X_test, y_test,
                                                                           best_params_list, nested_scores, outer_scorer)
                            classifiers_list.append(str(self.name)+'_full')
                            number_of_features.append(num_features)
                            way_of_selection.append('none')
                        else: 
                            raise ValueError('num_features must be an integer less than the number of features in the dataset')
                    elif type(num_features) is list :
                        for num_feature in num_features:
                            if num_feature > X_train.shape[1]:
                                raise ValueError('num_features must be less than the number of features in the dataset')
                            elif num_feature == X_train.shape[1]:
                                nested_scores_tr, best_params_list_tr = self.inner_loop(optimizer, inner_scoring, inner_cv, n_jobs, verbose,
                                                                               n_iter, n_trials,X_train, y_train, X_test, y_test,
                                                                               best_params_list, nested_scores, outer_scorer)
                                classifiers_list.append(str(self.name)+'_full')
                                number_of_features.append(num_feature)
                                way_of_selection.append('none')
                            else:
                                self.selected_features = mrmr_classif(X_train, y_train, K=num_feature)
                                X_train_selected = X_train[self.selected_features]
                                X_test_selected = X_test[self.selected_features]
                                nested_scores_tr, best_params_list_tr = self.inner_loop(optimizer, inner_scoring, inner_cv, n_jobs, verbose,
                                                                               n_iter, n_trials,X_train_selected, y_train, X_test_selected, y_test,
                                                                               best_params_list, nested_scores, outer_scorer)
                                classifiers_list.append(str(self.name)+'_'+str(num_feature)+'_'+str(feature_selection_type))
                                number_of_features.append(num_feature)
                                way_of_selection.append('none')
                    else: 
                        raise ValueError('num_features must be an integer or a list')

                else: 
                    nested_scores_tr, best_params_list_tr = self.inner_loop(optimizer, inner_scoring, inner_cv, n_jobs, verbose,
                                                                           n_iter, n_trials,X_train, y_train, X_test, y_test,
                                                                           best_params_list, nested_scores, outer_scorer)
                    classifiers_list.append(str(self.name)+'_full')
                    number_of_features.append(X_train.shape[1])
                    way_of_selection.append('full')
                j+=1
            # for score in range(len(nested_scores_tr)):
            #     nested_scores.append(nested_scores_tr[score])
            #     best_params_list.append(best_params_list_tr[score])
            #     classifiers_list.append(str(self.name)+'_'+str(num_features)+'_'+str(feature_selection_type))
                    # self.selected_features = mrmr_classif(self.X, self.y, K=num_features)
                    # self.X = self.X[self.selected_features]
                    # self.estimator = SelectFromModel(self.estimator, prefit=False, threshold=-np.inf, max_features=num_features)
                    # self.estimator.fit(X_train, y_train)
                    # X_train = self.estimator.transform(X_train)
                    # X_test = self.estimator.transform(X_test)
                
                # if optimizer == 'grid_search':
                #     clf = GridSearchCV(estimator=self.estimator, scoring=inner_scoring, 
                #                 param_grid=self.param_grid, cv=inner_cv, n_jobs=n_jobs, verbose=verbose)
                # elif optimizer == 'random_search':
                #     clf = RandomizedSearchCV(estimator=self.estimator, scoring=inner_scoring,
                #                              param_distributions=self.param_grid, cv=inner_cv, n_jobs=n_jobs, 
                #                              verbose=verbose, n_iter=n_iter)
                # elif optimizer == 'bayesian_search':
                #     clf = optuna.integration.OptunaSearchCV(estimator=self.estimator, scoring=inner_scoring,
                #                                         param_distributions=optuna_grid['NestedCV'][self.name], cv=inner_cv, n_jobs=n_jobs, 
                #                                         verbose=verbose, n_trials=n_trials)
                # else:
                #     raise Exception("Unsupported optimizer.")        

                # clf.fit(X_train, y_train)
                # best_params_list.append(clf.best_params_)
                # y_pred = clf.predict(X_test)
                # nested_score = outer_scorer(clf, X_test, y_test)
                # nested_scores.append(nested_score)
            
        df = pd.DataFrame({
            'Scores': nested_scores,
            'Best_Params': best_params_list,
            'Classifiers': classifiers_list,
            'Number_of_Features': number_of_features,
            'Way_of_Selection': way_of_selection})
        return df
    
    def model_selection(self,optimizer = 'grid_search', n_trials=100, n_iter=25, 
                        num_trials=10, score = 'matthews_corrcoef', exclude=None,
                        search_on=None, scores_df=False, alpha=1,num_features=None,
                        feature_selection_type='mrmr', plot='box' , train_best=None,
                        return_model=False,choose_model=False):
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
            print(f'Starting {estimator}...')
            self.name = estimator
            self.estimator = self.available_clfs[estimator]
            df = self.nested_cross_validation(optimizer=optimizer, n_trials=n_trials, n_iter=n_iter,
                                          num_trials=num_trials, inner_scoring=score, outer_scoring=score,
                                          feature_selection_type=feature_selection_type, num_features=num_features)
            
            for classif in np.unique(df['Classifiers']):
                print(f'Resulting {classif}...')
                indices = np.where(df['Classifiers'] == classif)[0]
                filtered_scores = df['Scores'][indices]
                filtered_best_params = df['Best_Params'][indices]
    
                mean_score = np.mean(filtered_scores)
                max_score = np.max(filtered_scores)
                std_score = np.std(filtered_scores)
                median_score = np.median(filtered_scores)
                evaluated_score = mean_score - alpha * std_score  
                Numbers_of_Features = np.unique(df['Number_of_Features'][indices])[0]
                Way_of_Selection = np.unique(df['Way_of_Selection'][indices])[0]

                results.append({
                    'Estimator': estimator,
                    'Classifier': classif,
                    'Scores': filtered_scores.tolist(),  
                    'Max': max_score,
                    'Std': std_score,
                    'Median': median_score,
                    'Evaluated': evaluated_score,
                    'Best_Params': filtered_best_params.tolist(),
                    'Numbers_of_Features': Numbers_of_Features,
                    'Way_of_Selection': Way_of_Selection 
                })
        
        scores_dataframe = pd.DataFrame(results) if scores_df else None
        # return scores_dataframe

        if plot == 'box':
            labels = scores_dataframe['Classifier'].unique() 
            plt.boxplot(scores_dataframe['Scores'], labels=labels)
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(rotation=90)  
            plt.grid(True)
            plt.show()
        elif plot == 'violin':
            plt.violinplot(all_scores)
            plt.title("Model Selection Results")
            plt.ylabel("Score")
            plt.xticks(range(1, len(clfs) + 1), scores_dataframe['Classifier'].unique(), rotation=90)
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

            features_selection = int(input("Enter the number for your preferred number of features: ")) - 1
            selected_features = features_options[features_selection]
            print(f"Selected classifier: {selected_classifier}, with {selected_features} features.")
            final_features = selected_features
            # if selected_features == 'all':
            #     print(f"Selected classifier: {selected_classifier}, with all features.")
            # else:
            #     print(f"Selected classifier: {selected_classifier}, with {selected_features} features.")
        else:
            self.name = max(results, key=lambda x: x['Evaluated'])['Estimator']
            final_features = max(results, key=lambda x: x['Evaluated'])['Numbers_of_Features']
            self.estimator = self.available_clfs[self.name] 


        # best_estimator_name = max(results, key=lambda x: x['Mean Score'])['Estimator']
        # self.name = max(results, key=lambda x: x['Evaluated'])['Estimator']
        # final_features = max(results, key=lambda x: x['Evaluated'])['Numbers_of_Features']
        # self.estimator = self.available_clfs[self.name]
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

        X2fit = self.X.copy()
        if num_features is not None:
            if feature_selection_type == 'mrmr':
                self.selected_features = mrmr_classif(X2fit, self.y, K=final_features)
                X2fit = X2fit[self.selected_features]
            elif feature_selection_type == 'kbest':
                X2fit = SelectKBest(chi2, k=final_features).fit_transform(X2fit, self.y)
            elif feature_selection_type == 'percentile':
                X2fit = SelectPercentile(chi2, percentile=final_features).fit_transform(X2fit, self.y)
            else:
                raise Exception("Unsupported feature selection method. Select one of 'mrmr', 'kbest', 'percentile'")                
        elif num_features is None or final_features == X2fit.shape[1]:
            pass

        if return_model and scores_df:
            self.best_estimator = self.estimator.fit(X2fit, self.y)
            return self.best_estimator, scores_dataframe
        elif return_model:
            self.best_estimator = self.estimator.fit(X2fit, self.y)
            return self.best_estimator
        elif scores_df:
            return scores_dataframe

            