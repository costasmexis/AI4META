import optuna
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import get_scorer, confusion_matrix, make_scorer
from sklearn.feature_selection import SelectFromModel

from .mlestimator import MachineLearningEstimator
from .optuna_grid import optuna_grid, hyper_compl
import os
import logging
from scipy.stats import sem
from itertools import chain
import time
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
import progressbar

import psycopg2
from psycopg2.extras import execute_values
import json
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from typing import Union
import copy
from datetime import datetime


def scoring_check(scoring: str) -> None:
    if (scoring not in sklearn.metrics.get_scorer_names()) and (scoring != "specificity"):
        raise ValueError(
            f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
        )


class MLPipelines(MachineLearningEstimator):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        super().__init__(label, csv_dir, estimator, param_grid)
        self.config_rncv = {}

    def _specificity_scorer(self, estimator, X, y):
        y_pred = estimator.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp)
        return specificity
    def _set_result_csv_name(self, csv_dir):
        """This function is used to set the name of the result nested cv file with respect to the dataset name"""
        data_name = os.path.basename(csv_dir).split(".")[0]
        return data_name
    
    def _bootstrap_ci(self, data, type='median'):
        ms = []
        for _ in range(1000):
            sample = np.random.choice(data, size=len(data), replace=True)
            if type == 'median':
                ms.append(np.median(sample))
            elif type == 'mean':
                ms.append(np.mean(sample))
        lower_bound = np.percentile(ms, (1 - 0.95) / 2 * 100)
        upper_bound = np.percentile(ms, (1 + 0.95) / 2 * 100)
        return lower_bound, upper_bound

    def _create_model_instance(self, model_name, params):
        """This function creates a model instance with the given parameters
        It is used in order to prevent fittinf of an already fitted model from previous runs"""

        if model_name == "RandomForestClassifier":
            if params == None:
                return RandomForestClassifier()
            else:
                return RandomForestClassifier(**params)
        elif model_name == "LogisticRegression":
            if params == None:
                return LogisticRegression()
            else:
                return LogisticRegression(**params)
        elif model_name == "XGBClassifier":
            if params == None:
                return XGBClassifier()
            else:
                return XGBClassifier(**params)
        elif model_name == "LGBMClassifier":
            if params == None:
                return LGBMClassifier(verbose=-1)
            else:
                return LGBMClassifier(**params)
        elif model_name == "CatBoostClassifier":
            if params == None:
                return CatBoostClassifier(verbose=0)
            else:
                return CatBoostClassifier(**params)
        elif model_name == "SVC":
            if params == None:
                return SVC()
            else:
                return SVC(**params)
        elif model_name == "KNeighborsClassifier":
            if params == None:
                return KNeighborsClassifier()
            else:
                return KNeighborsClassifier(**params)
        elif model_name == "LinearDiscriminantAnalysis":
            if params == None:
                return LinearDiscriminantAnalysis()
            else:
                return LinearDiscriminantAnalysis(**params)
        elif model_name == "GaussianNB":
            if params == None:
                return GaussianNB()
            else:
                return GaussianNB(**params)
        elif model_name == "GradientBoostingClassifier":
            if params == None:
                return GradientBoostingClassifier()
            else:
                return GradientBoostingClassifier(**params)
        elif model_name == "GaussianProcessClassifier":
            if params == None:
                return GaussianProcessClassifier()
            else:
                return GaussianProcessClassifier(**params)
        elif model_name == "ElasticNet":
            if params == None:
                return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5)
            else:
                return LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _gso_model(self, trials, model_name, splits, method):
        """This function selects the 'simplest' hyperparameters for the given model."""
        trials_data = [
            {
                "params": t.params,
                "mean_test_score":  t.user_attrs.get('mean_test_score'),
                "mean_train_score": t.user_attrs.get('mean_train_score'),
                "test_scores": [t.user_attrs.get(f"split{i}_test_score") for i in range(splits)],
                "train_scores": [t.user_attrs.get(f"split{i}_train_score") for i in range(splits)],
                "gap_scores": [np.abs(t.user_attrs.get(f"split{i}_test_score") - t.user_attrs.get(f"split{i}_train_score")) for i in range(splits)]
            }
            for t in trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        
        if method == "gso_1":
            # Sort trials by mean train score
            trials_data = sorted(
                        trials_data, key=lambda x: (x["mean_train_score"]), reverse=True
                    )

            # Find the best train score and set a threshold
            best_train_score = trials_data[0]["mean_train_score"]
            k = 0.85
            train_score_threshold = k * best_train_score

            # Filter trials by those that are above the test score threshold and have a train score not lower than the test score
            filtered_trials = [t for t in trials_data if (t["mean_train_score"] >= train_score_threshold)]

            # Select the trial with the smallest average gap score
            if filtered_trials:
                gso_1_trial = min(filtered_trials, key=lambda x: np.mean(x["gap_scores"]))
                return gso_1_trial["params"]
            else:
                return trials[0].params
            
        elif method == "gso_2":
            # Sort trials by mean test score
            trials_data = sorted(
                trials_data, key=lambda x: (x["mean_test_score"]), reverse=True
            )

            # Find the best validation score and set a threshold
            best_test_score = trials_data[0]["mean_test_score"]
            k = 0.85
            test_score_threshold = k * best_test_score

            # Filter trials by those that are above the test score threshold and have a train score not lower than the test score
            filtered_trials = [t for t in trials_data if (t["mean_train_score"] >= test_score_threshold)]

            # Select the trial with the smallest average gap score
            if filtered_trials:
                gso_1_trial = min(filtered_trials, key=lambda x: np.mean(x["gap_scores"]))
                return gso_1_trial["params"]
            else:
                return trials[0].params

    def _one_sem_model(self, trials, model_name, samples, splits, method):
        """This function selects the 'simplest' hyperparameters for the given model."""
        constraints = hyper_compl[model_name]
        
        trials_data = [
            {
                "params": t.params,
                "value": t.values[0],
                "sem": t.user_attrs["std_test_score"] / (splits**0.5),
                "train_time": t.user_attrs["mean_fit_time"],
            }
            for t in trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trials_data = sorted(
            trials_data, key=lambda x: (x["value"], -x["sem"]), reverse=True
        )

        # Find the best score and its SEM value
        best_score = trials_data[0]["value"]
        best_sem_score = trials_data[0]["sem"]

        # Find the scores that will possibly return simpler models with equally good performance
        sem_threshold = best_score - best_sem_score
        filtered_trials = [t for t in trials_data if t["value"] >= sem_threshold]
        if method == "one_sem":
            def calculate_complexity(trial, model_name, samples):
                params = trial["params"]
                if model_name in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier', 'LGBMClassifier']:
                    max_depth = params["max_depth"]
                    if model_name == 'RandomForestClassifier' or model_name == 'GradientBoostingClassifier':
                        actual_depth = min((samples / params["min_samples_leaf"]), max_depth)
                    elif model_name == 'XGBClassifier':
                        actual_depth = max_depth  # Assuming XGBClassifier does not use min_samples_leaf
                    elif model_name == 'LGBMClassifier':
                        actual_depth = min((samples / params["min_child_samples"]), max_depth)
                    complexity = params["n_estimators"] * (2 ** (actual_depth - 1))
                elif model_name == 'CatBoostClassifier':
                    max_depth = params["depth"]
                    actual_depth = min((samples / params["min_data_in_leaf"]), max_depth)
                    complexity = params["n_estimators"] * (2 ** (actual_depth - 1))#*params["iterations"]
                elif model_name == 'LogisticRegression' or model_name == 'ElasticNet':
                    complexity = params["C"] * params["max_iter"]
                elif model_name == 'SVC':
                    if params["kernel"] == 'poly':
                        complexity = params["C"] * params["degree"]
                    else:
                        complexity = params["C"]
                elif model_name == 'KNeighborsClassifier':
                    complexity = params["leaf_size"]
                elif model_name == 'LinearDiscriminantAnalysis':
                    complexity = params["shrinkage"]
                elif model_name == 'GaussianNB':
                    complexity = params["var_smoothing"]
                elif model_name == 'GaussianProcessClassifier':
                    complexity = params["max_iter_predict"]*params["n_restarts_optimizer"]
                else:
                    complexity = float('inf')  # If model not recognized, set high complexity
                return complexity

            # Calculate complexity for each filtered trial
            for trial in filtered_trials:
                trial["complexity"] = calculate_complexity(trial, model_name, samples)

            # Find the trial with the smallest complexity
            shorted_trials = sorted(filtered_trials, key=lambda x: (x["complexity"], x["train_time"]))
            best_trial = shorted_trials[0]

            return best_trial["params"]
        elif method == "one_sem_grd":
            # Retrieve the hyperparameter priorities for the given model type
            hyperparams = hyper_compl[model_name]

            # Iterate over the hyperparameters and their sorting orders
            for hyper, order in hyperparams.items():
                # Sort the models based on the current hyperparameter
                sorted_dict = sorted(filtered_trials, key=lambda x: x['params'][hyper], reverse=not order)

                # Get the best value for the current hyperparameter from the sorted list
                best_value = sorted_dict[0]['params'][hyper]

                # Find all models with the best value for the current hyperparameter
                models_with_same_hyper = []
                for model in sorted_dict:
                    if model['params'][hyper] == best_value:
                        models_with_same_hyper.append(model)

                # If there is only one model with the best hyperparameter value, return it
                if len(models_with_same_hyper) == 1:
                    filtered_trials = [models_with_same_hyper[0]].copy()
                    break
                else:
                    # Otherwise, update all_models to only include models with the best hyperparameter value
                    filtered_trials = models_with_same_hyper.copy()

            # If multiple models have the same best hyperparameter values, return the first one
            simple_model = filtered_trials[0]

            return simple_model["params"]
          
    def _sfm(self, estimator, X_train, X_test, y_train, num_feature2_use=None, threshold="mean"):
        """
        Select features using SelectFromModel with either a predefined number of features 
        or using the threshold if num_feature2_use is not provided.
        """

        # Ensure the model is fitted
        estimator.fit(X_train, y_train)
        if  num_feature2_use is None:
            # Create the SelectFromModel object using the provided threshold
            sfm = SelectFromModel(estimator, threshold=threshold)
        else: 
            sfm = SelectFromModel(estimator, max_features=num_feature2_use)

        # Fit the model to select important features
        sfm.fit(X_train, y_train)
        
        # get a boolean list the selected features
        selected_features = sfm.get_support(indices=True)
        selected_columns = X_train.columns[selected_features].to_list()
        
        # Select the features based on either the threshold or top num_feature2_use
        X_train_selected = X_train[selected_columns]
        X_test_selected = X_test[selected_columns]

        return X_train_selected, X_test_selected, num_feature2_use
    
    def _cv_loop(self, i, avail_thr):
        start = time.time()  # Count time of outer loops
        
        # Split the data into train and test
        self.config_rcv["cv_splits"] = StratifiedKFold(
            n_splits=self.config_rcv["splits"], shuffle=True, random_state=i
        )

        train_test_indices = list(self.config_rcv["cv_splits"].split(self.X, self.y))

        # Store the results in a list od dataframes
        list_dfs = []

        # Initiate the progress bar
        widgets = [
            progressbar.Percentage(),
            " ",
            progressbar.GranularBar(),
            " ",
            progressbar.Timer(),
            " ",
            progressbar.ETA(),
        ]

        temp_list = []
        with progressbar.ProgressBar(
            prefix=f"Round {i+1} of CV:",
            max_value=self.config_rcv["splits"],
            widgets=widgets,
        ) as bar:
            split_index = 0

            # For each outer fold perform
            for train_index, test_index in train_test_indices:

                # Checks for reliability of parameters
                if isinstance(self.config_rcv["num_features"], int):
                    feature_loop = [self.config_rcv["num_features"]]
                elif isinstance(self.config_rcv["num_features"], list):
                    feature_loop = self.config_rcv["num_features"]
                elif self.config_rcv["num_features"] is None:
                    feature_loop = [self.X.shape[1]]
                else:
                    raise ValueError("num_features must be an integer or a list or None")

                # Initialize variables
                results = {
                    "Classifiers": [],
                    "Selected_Features": [],
                    "Number_of_Features": [],
                    "Way_of_Selection": [],
                    "Estimator": [],
                    'Samples_counts': [],
                }
                results.update({f"{metric}": [] for metric in self.config_rcv["extra_metrics"]})

                # Fold over the number of features
                for num_feature2_use in feature_loop:
                    if not self.config_rcv['sfm']:
                        X_train_selected, X_test_selected, num_feature = self._filter_features(
                            train_index, test_index, self.X, self.y, num_feature2_use, cvncvsel='rcv'
                        )
                        y_train, y_test = self.y[train_index], self.y[test_index]
                        
                        # Apply class balancing strategies
                        if self.config_rcv['class_balance'] == 'auto':
                            # No balancing is applied; use original X_train_selected and y_train
                            pass
                        elif self.config_rcv['class_balance'] == 'smote':
                            X_train_selected, y_train = SMOTE(random_state=i).fit_resample(X_train_selected, y_train)
                        elif self.config_rcv['class_balance'] == 'smote_enn':
                            X_train_selected, y_train = SMOTEENN(random_state=i, enn=EditedNearestNeighbours()).fit_resample(X_train_selected, y_train)
                        elif self.config_rcv['class_balance'] == 'adasyn':
                            X_train_selected, y_train = ADASYN(random_state=i ).fit_resample(X_train_selected, y_train)
                        elif self.config_rcv['class_balance'] == 'borderline_smote':
                            X_train_selected, y_train = BorderlineSMOTE(random_state=i).fit_resample(X_train_selected, y_train)
                        elif self.config_rcv['class_balance'] == 'tomek':
                            tomek = TomekLinks()
                            X_train_selected, y_train = tomek.fit_resample(X_train_selected, y_train)

                    # Check of the classifiers given list
                    if self.config_rcv["clfs"] is None:
                        raise ValueError("No classifier specified.")
                    else:
                        for estimator in self.config_rcv["clfs"]:
                            self.name = estimator
                            self.estimator = self.available_clfs[estimator]
                            if (self.config_rcv["sfm"]) and ((estimator == "RandomForestClassifier") or 
                                        (estimator == "XGBClassifier") or 
                                        (estimator == 'GradientBoostingClassifier') or 
                                        (estimator == "LGBMClassifier") or 
                                        (estimator == "CatBoostClassifier")):
                                
                                X_train_selected, X_test_selected, num_feature = self._filter_features(
                                    train_index, test_index, self.X, self.y, num_feature2_use=self.X.shape[1], cvncvsel='rcv'
                                )
                                y_train, y_test = self.y[train_index], self.y[test_index]

                                # Perform feature selection using Select From Model (sfm)
                                X_train_selected, X_test_selected, num_feature = self._sfm(self.estimator, X_train_selected, X_test_selected, y_train, num_feature2_use)

                                # Apply class balancing strategies after sfm
                                if self.config_rcv['class_balance'] == 'auto':
                                    # No balancing is applied; use original X_train_selected and y_train
                                    pass
                                elif self.config_rcv['class_balance'] == 'smote':
                                    X_train_selected, y_train = SMOTE(random_state=i).fit_resample(X_train_selected, y_train)
                                elif self.config_rcv['class_balance'] == 'smote_enn':
                                    X_train_selected, y_train = SMOTEENN(random_state=i, enn=EditedNearestNeighbours()).fit_resample(X_train_selected, y_train)
                                elif self.config_rcv['class_balance'] == 'adasyn':
                                    X_train_selected, y_train = ADASYN(random_state=i).fit_resample(X_train_selected, y_train)
                                elif self.config_rcv['class_balance'] == 'borderline_smote':
                                    X_train_selected, y_train = BorderlineSMOTE(random_state=i).fit_resample(X_train_selected, y_train)
                                elif self.config_rcv['class_balance'] == 'tomek':
                                    tomek = TomekLinks()
                                    X_train_selected, y_train = tomek.fit_resample(X_train_selected, y_train)

                            # Train the model
                            clf = self._create_model_instance(
                                    self.name, params=None
                                )
                            clf.fit(X_train_selected, y_train)
                            
                            # Store the results and apply one_sem method if its selected
                            results["Estimator"].append(self.name)
                            for metric in self.config_rcv["extra_metrics"]:
                                if metric == 'specificity':
                                    results[f"{metric}"].append(
                                        self._specificity_scorer(clf, X_test_selected, y_test)
                                    )
                                else:
                                    # For all other metrics, use get_scorer
                                    results[f"{metric}"].append(
                                        get_scorer(metric)(clf, X_test_selected, y_test)
                                    )
                            y_pred = clf.predict(X_test_selected)

                            # Store the results using different names if feature selection is applied
                            if num_feature == "full" or num_feature is None:
                                results["Selected_Features"].append(None)
                                results["Number_of_Features"].append(X_test_selected.shape[1])
                                results["Way_of_Selection"].append("full")
                                results["Classifiers"].append(f"{self.name}")
                            else:
                                results["Classifiers"].append(
                                    f"{self.name}_{self.config_rcv['feature_selection_type']}_{num_feature}"
                                )
                                results["Selected_Features"].append(
                                    X_train_selected.columns.tolist()
                                )
                                results["Number_of_Features"].append(num_feature)
                                if (self.config_rcv["sfm"]) and ((estimator == "RandomForestClassifier") or 
                                        (estimator == "XGBClassifier") or
                                        (estimator == 'GradientBoostingClassifier') or
                                        (estimator == "LGBMClassifier") or
                                        (estimator == "CatBoostClassifier")):
                                    results["Way_of_Selection"].append(
                                        'sfm'
                                    )
                                else:
                                    results["Way_of_Selection"].append(
                                        self.config_rcv["feature_selection_type"]
                                    )

                            # Track predictions
                            samples_counts = np.zeros(len(self.y))
                            for idx, resu, pred in zip(test_index, y_test, y_pred):
                                if pred == resu:
                                    samples_counts[idx] += 1

                            results['Samples_counts'].append(samples_counts)

                temp_list.append([results])
                bar.update(split_index)
                split_index += 1
                time.sleep(1)

            list_dfs = [item for sublist in temp_list for item in sublist]
            end = time.time()

        # Return the list of dataframes and the time fold
        print(f"Finished with {i+1} round after {(end-start)/3600:.2f} hours.")
        return list_dfs 

    def rcv_accel(        
        self,
        rounds=10,
        exclude=None,
        search_on=None,
        num_features=None,
        feature_selection_type="mrmr",
        return_csv=True,
        feature_selection_method="chi2",
        plot="box",
        scoring="matthews_corrcoef",
        splits=5,
        freq_feat=None,
        normalization="minmax",
        missing_values_method="median",
        name_add=None,
        class_balance = 'auto',
        sfm=False,
        extra_metrics=['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','average_precision','specificity'],
        info_to_db=False,
        filter_csv=None
    ):
        # Missing values manipulation
        if missing_values_method == "drop":
            print(
                "Values cannot be dropped at ncv because of inconsistent shapes. \nThe missing values with automaticly replaced by the median of each feature."
            )
            missing_values_method = "median"
            
        if self.X.isnull().values.any():
            print(
                f"Your Dataset contains NaN values. Some estimators does not work with NaN values.\nThe {missing_values_method} method will be used for the missing values manipulation.\n"
            )
            
        if extra_metrics is not None:
            if type(extra_metrics) is not list:
                extra_metrics = [extra_metrics]
            for metric in extra_metrics:
                scoring_check(metric)
            print('All the extra metrics are valid.')
            if scoring not in extra_metrics:
                extra_metrics.insert(0, scoring)
            elif scoring in extra_metrics and extra_metrics.index(scoring) != 0:
                # Remove it from its current position
                extra_metrics.remove(scoring)
                # Insert it at the first index
                extra_metrics.insert(0, scoring)
        else:
            extra_metrics = [scoring]
            
        if class_balance not in ['auto','smote','smote_enn','adasyn','borderline_smote','tomek', None]:
            raise ValueError("class_balance must be one of the following: 'auto','smote','smotenn','adasyn','borderline_smote','tomek', or None")
        elif class_balance == None:
            class_balance = 'auto'
            print('Class balance is set to "auto"')
            
        # Set parameters for the nested functions of the cv process
        self.config_rcv = locals()
        self.config_rcv.pop("self", None)
        self.config_rcv['dataset_name'] = self.csv_dir
        self.config_rcv['model_selection_type'] = 'rcv'

        if num_features is not None:
            print(
                f"The num_features parameter is {num_features}."#\nThe result will be a Dataframe and a List with the freq_feat number of the most important features.\nIf the freq_feat is None, the result will be a List with all features."
            )
            
        # Set available classifiers
        if exclude is not None:
            exclude_classes = (
                exclude  # 'exclude' is a list of classifier names as strings
            )
        elif search_on is not None:
            classes = search_on  # 'search_on' is a list of classifier names as strings
            exclude_classes = [
                clf for clf in self.available_clfs.keys() if clf not in classes
            ]
        else:
            exclude_classes = []

        # Filter classifiers based on the exclude_classes list
        clfs = [clf for clf in self.available_clfs.keys() if clf not in exclude_classes]
        self.config_rcv["clfs"] = clfs

        # Checks for reliability of parameters
        if (scoring not in sklearn.metrics.get_scorer_names()) and (scoring != "specificity"):
            raise ValueError(
                f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and 'specificity'"
            )

        # Parallelization
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        if num_cores < rounds:
            use_cores = num_cores
        else:
            use_cores = rounds
        avail_thr = max(1, num_cores // rounds)

        with threadpool_limits():
            list_dfs = Parallel(n_jobs=use_cores,verbose=0)(
                delayed(self._cv_loop)(i,avail_thr) for i in trial_indices
            )

        list_dfs_flat = list(chain.from_iterable(list_dfs))
        
         # Create results dataframe
        results = []
        df = pd.DataFrame()
        for item in list_dfs_flat:
            dataframe = pd.DataFrame(item)
            df = pd.concat([df, dataframe], axis=0)

        for classif in np.unique(df["Classifiers"]):
            indices = df[df["Classifiers"] == classif]
            filtered_scores = indices[f"{self.config_rcv['scoring']}"].values
            if num_features is not None:
                filtered_features = indices["Selected_Features"].values
            mean_score = np.mean(filtered_scores)
            max_score = np.max(filtered_scores)
            std_score = np.std(filtered_scores)
            sem_score = sem(filtered_scores)
            median_score = np.median(filtered_scores)
            Numbers_of_Features = indices["Number_of_Features"].unique()[0]
            Way_of_Selection = indices["Way_of_Selection"].unique()[0]
            samples_classification_rates = np.zeros(len(self.y))
            for test_part in indices["Samples_counts"]:
                samples_classification_rates=np.add(samples_classification_rates,test_part)
            samples_classification_rates /= rounds
                            
            results.append(
                {
                    "Est": df[df["Classifiers"] == classif]["Estimator"].unique()[
                        0
                    ],
                    "Clf": classif,
                    "Hyp": 'Default',
                    "Sel_feat": filtered_features
                    if num_features is not None
                    else None,
                    "Fs_num": Numbers_of_Features,
                    "Sel_way": Way_of_Selection,
                    "Fs_inner": feature_selection_method,
                    "Norm": normalization,
                    "Miss_vals": missing_values_method,
                    "Splits": splits,
                    "Rnds": rounds,
                    "Class_bal": class_balance,
                    "Scoring": scoring,
                    "In_sel": 'validation_score',
                    "Classif_rates": samples_classification_rates.tolist(),
                }
            )
            
            results = self._input_renamed_metrics(
                extra_metrics, results, indices
            )
                                            
        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)

        # Create a 'Results' directory
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Initiate name
        final_dataset_name = self._name_outputs(self.config_rcv, results_dir)  
          
        # Save the results to a CSV file of the outer scores for each classifier
        if return_csv:
            statistics_dataframe = self._return_csv(final_dataset_name, scores_dataframe, extra_metrics, filter_csv)

        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:
            # Plot histogram of features
            self._histogram(scores_dataframe, final_dataset_name, freq_feat, clfs)
        
        # Plot box or violin plots of the outer cross-validation scores 
        if plot is not None:
            self._plot(scores_dataframe, plot, self.config_rcv['scoring'], final_dataset_name)
            
        if info_to_db:
            # Add to database
            self._insert_data_into_db(scores_dataframe, self.config_rcv)

        return statistics_dataframe

    def _plot(self, scores_dataframe, plot, scorer, final_dataset_name):
        
        scores_long = scores_dataframe.explode(f"{scorer}")
        scores_long[f"{scorer}"] = scores_long[f"{scorer}"].astype(float)
        fig = go.Figure()
        
        classifiers = scores_long["Clf"].unique()

        if plot == "box":
            # Add box plots for each classifier within each Inner_Selection method
            for classifier in classifiers:
                data = scores_long[scores_long["Clf"] == classifier][
                    f"{scorer}"
                ]
                median = np.median(data)
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=f"{classifier} (Median: {median:.2f})",
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    )
                )

                # Calculate and add 95% CI for the median
                lower, upper = self._bootstrap_ci(data, type='median')
                fig.add_trace(
                    go.Scatter(
                        x=[f"{classifier} (Median: {median:.2f})",
                        f"{classifier} (Median: {median:.2f})"],
                        y=[lower, upper],
                        mode="lines",
                        line=dict(color="black", dash="dash"),
                        showlegend=False,
                    )
                )

        elif plot == "violin":
            for classifier in classifiers:
                data = scores_long[scores_long["Clf"] == classifier][
                    f"{scorer}"
                ]
                median = np.median(data)
                fig.add_trace(
                    go.Violin(
                        y=data,
                        name=f"{classifier} (Median: {median:.2f})",
                        box_visible=False,
                        points="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    )
                )
        else:
            raise ValueError(
                f'The "{plot}" is not a valid option for plotting. Choose between "box" or "violin".'
            )

        # Update layout for better readability
        fig.update_layout(
            autosize = False,
            width=1500,
            height=1200,
            title="Model Selection Results by Classifier",
            yaxis_title=f"Scores {scorer}",
            xaxis_title="Classifier",
            xaxis_tickangle=-45,
            template="plotly_white",
        )
        
        # Save the figure as an image in the "Results" directory
        image_path = f"{final_dataset_name}_model_selection_plot.png"
        fig.write_image(image_path)
              
    def _histogram(self, scores_dataframe, final_dataset_name, freq_feat, clfs):
        if freq_feat == None:
            freq_feat = self.X.shape[1]
        elif freq_feat > self.X.shape[1]:
            freq_feat = self.X.shape[1]
            
        # Plot histogram of features
        feature_counts = Counter()
        for idx, row in scores_dataframe.iterrows():
            if row["Sel_way"] != "full":  # If no features were selected, skip
                features = list(
                    chain.from_iterable(
                        [list(index_obj) for index_obj in row["Sel_feat"]]
                    )
                )
                feature_counts.update(features)

        sorted_features_counts = feature_counts.most_common()

        if len(sorted_features_counts) == 0:
            print("No features were selected.")
        else:
            features, counts = zip(*sorted_features_counts[:freq_feat])
            counts = [x / len(clfs) for x in counts]  # Normalize counts
            print(f"Selected {freq_feat} features")

            # Create the bar chart using Plotly
            fig = go.Figure()

            # Add bars to the figure
            fig.add_trace(go.Bar(
                x=features,
                y=counts,
                marker=dict(color="skyblue"),
                text=[f"{count:.2f}" for count in counts],  # Show normalized counts as text
                textposition='auto'
            ))

            # Set axis labels and title
            fig.update_layout(
                title="Histogram of Selected Features",
                xaxis_title="Features",
                yaxis_title="Counts",
                xaxis_tickangle=-90,  # Rotate x-ticks to avoid overlap
                bargap=0.2,
                template="plotly_white",
                width=min(max(1000, freq_feat * 20), 2000),  # Dynamically adjust plot width
                height=700  # Set plot height
            )

            # Save the plot to 'Results/histogram.png'
            save_path = f"{final_dataset_name}_histogram.png"
            fig.write_image(save_path)
            
    def _name_outputs(self, config, results_dir):
        try:
            dataset_name = self._set_result_csv_name(self.csv_dir)
            name_add  = self._file_name(config)
            results_name = f"{dataset_name}_{name_add}_{config['model_selection_type']}"
            final_dataset_name = os.path.join(results_dir, results_name)
        except Exception as e:
            name_add = self._file_name(config)
            results_name = f"results_{name_add}_{config['model_selection_type']}"
            final_dataset_name = os.path.join(
                results_dir, results_name
            )
        return final_dataset_name
            
    def _filter_features(self, train_index, test_index, X, y, num_feature2_use, cvncvsel):
        """
        This function filters the features using the selected model.
        """
        if cvncvsel == "rcv":
            config = self.config_rcv
        else:
            config = self.config_rncv
        
        # Find the train and test sets
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y[train_index], y[test_index]

        X_train, X_test = self.normalize(
            X=X_tr,
            train_test_set=True,
            X_test=X_te,
            method=config["normalization"],
        )

        # Manipulate the missing values for both train and test sets
        X_train = self.missing_values(
            data=X_train, method=config["missing_values_method"]
        )
        X_test = self.missing_values(
            data=X_test, method=config["missing_values_method"]
        )

        # Find the feature selection type and apply it to the train and test sets
        if config["feature_selection_type"] != "percentile":
            if isinstance(num_feature2_use, int):
                if num_feature2_use < X_train.shape[1]:
                    self.selected_features = self.feature_selection(
                        X=X_train,
                        y=y_train,
                        method=config["feature_selection_type"],
                        num_features=num_feature2_use,
                        inner_method=config["feature_selection_method"],
                    )
                    X_train_selected = X_train[self.selected_features]
                    X_test_selected = X_test[self.selected_features]
                    num_feature = num_feature2_use
                elif num_feature2_use == X_train.shape[1]:
                    X_train_selected = X_train
                    X_test_selected = X_test
                    num_feature = "full"
                else:
                    raise ValueError(
                        "num_features must be an integer less than the number of features in the dataset"
                    )
        elif config["feature_selection_type"] == "percentile":
            if isinstance(num_feature2_use, int):
                if (
                    num_feature2_use == 100 or num_feature2_use == self.X.shape[1]
                ):  # TODO: check how to write it better
                    X_train_selected = X_train
                    X_test_selected = X_test
                    num_feature = "full"
                elif num_feature2_use < 100 and num_feature2_use > 0:
                    self.selected_features = self.feature_selection(
                        X=X_train,
                        y=y_train,
                        method=config["feature_selection_type"],
                        inner_method=config["feature_selection_method"],
                        num_features=num_feature2_use,
                    )
                    X_train_selected = X_train[self.selected_features]
                    X_test_selected = X_test[self.selected_features]
                    num_feature = num_feature2_use
                else:
                    raise ValueError(
                        "num_features must be an integer less or equal than 100 and hugher thatn 0"
                    )
        else:
            raise ValueError("num_features must be an integer or a list or None")

        return X_train_selected, X_test_selected, num_feature

    def _inner_loop(self, train_index, test_index, X, y, avail_thr, i):
        """This function is used to perform the inner loop of the nested cross-validation
        Note: Return a list because this is the desired output for the parallel loop
        """
        opt_grid = "NestedCV"

        # Checks for reliability of parameters
        if isinstance(self.config_rncv["num_features"], int):
            feature_loop = [self.config_rncv["num_features"]]
        elif isinstance(self.config_rncv["num_features"], list):
            feature_loop = self.config_rncv["num_features"]
        elif self.config_rncv["num_features"] is None:
            feature_loop = [X.shape[1]]
        else:
            raise ValueError("num_features must be an integer or a list or None")

        if self.config_rncv["parallel"] == "thread_per_round":
            n_jobs = 1
        elif self.config_rncv["parallel"] == "freely_parallel":
            n_jobs = avail_thr

        # Initialize variables
        results = {
            "Classifiers": [],
            "Selected_Features": [],
            "Number_of_Features": [],
            "Hyperparameters": [],
            "Way_of_Selection": [],
            "Estimator": [],
            'Samples_counts': [],
            'Inner_selection_mthd': [],
        }
        results.update({f"{metric}": [] for metric in self.config_rncv["extra_metrics"]})

        # Loop over the number of features
        for num_feature2_use in feature_loop:            
            # if not self.config_rncv['sfm']:
            X_train_selected, X_test_selected, num_feature = self._filter_features(
                train_index, test_index, X, y, num_feature2_use, cvncvsel='rncv'
            )
            y_train, y_test = y[train_index], y[test_index]
            
            # Apply class balancing strategies
            if self.config_rncv['class_balance'] == 'auto':
                # No balancing is applied; use original X_train_selected and y_train
                pass
            elif self.config_rncv['class_balance'] == 'smote':
                X_train_selected, y_train = SMOTE(random_state=i, n_jobs=n_jobs).fit_resample(X_train_selected, y_train)
            elif self.config_rncv['class_balance'] == 'smote_enn':
                X_train_selected, y_train = SMOTEENN(random_state=i, n_jobs=n_jobs, enn=EditedNearestNeighbours()).fit_resample(X_train_selected, y_train)
            elif self.config_rncv['class_balance'] == 'adasyn':
                X_train_selected, y_train = ADASYN(random_state=i, n_jobs=n_jobs).fit_resample(X_train_selected, y_train)
            elif self.config_rncv['class_balance'] == 'borderline_smote':
                X_train_selected, y_train = BorderlineSMOTE(random_state=i, n_jobs=n_jobs).fit_resample(X_train_selected, y_train)
            elif self.config_rncv['class_balance'] == 'tomek':
                tomek = TomekLinks(n_jobs=n_jobs)
                X_train_selected, y_train = tomek.fit_resample(X_train_selected, y_train)
                                
            # Check of the classifiers given list
            if self.config_rncv["clfs"] is None:
                raise ValueError("No classifier specified.")
            else:
                for estimator in self.config_rncv["clfs"]:
                    # For every estimator find the best hyperparameteres
                    self.name = estimator
                    self.estimator = self.available_clfs[estimator]
                    if (self.config_rncv["sfm"]) and ((estimator == "RandomForestClassifier") or 
                                (estimator == "XGBClassifier") or 
                                (estimator == 'GradientBoostingClassifier') or 
                                (estimator == "LGBMClassifier") or 
                                (estimator == "CatBoostClassifier")):
                        
                        X_train_selected, X_test_selected, num_feature = self._filter_features(
                            train_index, test_index, X, y, num_feature2_use=X.shape[1], cvncvsel='rncv'
                        )
                        y_train, y_test = y[train_index], y[test_index]

                        # Perform feature selection using Select From Model (sfm)
                        X_train_selected, X_test_selected, num_feature = self._sfm(self.estimator, X_train_selected, X_test_selected, y_train, num_feature2_use)

                        # Apply class balancing strategies after sfm
                        if self.config_rncv['class_balance'] == 'auto':
                            # No balancing is applied; use original X_train_selected and y_train
                            pass
                        elif self.config_rncv['class_balance'] == 'smote':
                            X_train_selected, y_train = SMOTE(random_state=i, n_jobs=n_jobs).fit_resample(X_train_selected, y_train)
                        elif self.config_rncv['class_balance'] == 'smote_enn':
                            X_train_selected, y_train = SMOTEENN(random_state=i, n_jobs=n_jobs, enn=EditedNearestNeighbours()).fit_resample(X_train_selected, y_train)
                        elif self.config_rncv['class_balance'] == 'adasyn':
                            X_train_selected, y_train = ADASYN(random_state=i, n_jobs=n_jobs).fit_resample(X_train_selected, y_train)
                        elif self.config_rncv['class_balance'] == 'borderline_smote':
                            X_train_selected, y_train = BorderlineSMOTE(random_state=i, n_jobs=n_jobs).fit_resample(X_train_selected, y_train)
                        elif self.config_rncv['class_balance'] == 'tomek':
                            tomek = TomekLinks(n_jobs=n_jobs)
                            X_train_selected, y_train = tomek.fit_resample(X_train_selected, y_train)
                            
                    self._set_optuna_verbosity(logging.ERROR)
                    clf = optuna.integration.OptunaSearchCV(
                        estimator=self.estimator,
                        scoring=self.config_rncv["inner_scoring"],
                        param_distributions=optuna_grid[opt_grid][self.name],
                        cv=self.config_rncv["inner_cv"],
                        return_train_score=True,
                        n_jobs=n_jobs,
                        verbose=0,
                        n_trials=self.config_rncv["n_trials"],
                    )
                    clf.fit(X_train_selected, y_train)
           
                    for inner_selection in self.config_rncv["inner_selection_lst"]:
                        results['Inner_selection_mthd'].append(inner_selection)
                        # Store the results and apply one_sem method if its selected
                        results["Estimator"].append(self.name)
                        if inner_selection == "validation_score":
                            
                            res_model = copy.deepcopy(clf)

                            params = res_model.best_params_

                        else:
                            trials = clf.trials_
                            if (inner_selection == "one_sem") or (inner_selection == "one_sem_grd"):
                                samples = X_train_selected.shape[0]
                                # Find simpler parameters with the one_sem method if there are any
                                simple_model_params = self._one_sem_model(trials, self.name, samples, self.config_rncv['inner_splits'],inner_selection)
                            elif (inner_selection == "gso_1") or (inner_selection == "gso_2"):
                                # Find parameters with the smaller gap score with gso_1 method if there are any
                                simple_model_params = self._gso_model(trials, self.name, self.config_rncv['inner_splits'],inner_selection)

                            params = simple_model_params

                            # Fit the new model
                            new_params_clf = self._create_model_instance(
                                self.name, simple_model_params
                            )
                            new_params_clf.fit(X_train_selected, y_train)

                            res_model = copy.deepcopy(new_params_clf)
                            
                        results["Hyperparameters"].append(params)
                        
                        for metric in self.config_rncv["extra_metrics"]:
                            if metric == 'specificity':
                                results[f"{metric}"].append(
                                    self._specificity_scorer(res_model, X_test_selected, y_test)
                                )
                            else:
                                results[f"{metric}"].append(
                                    get_scorer(metric)(res_model, X_test_selected, y_test)
                                )

                        y_pred = res_model.predict(X_test_selected)

                        # Store the results using different names if feature selection is applied
                        if num_feature == "full" or num_feature is None:
                            results["Selected_Features"].append(None)
                            results["Number_of_Features"].append(X_test_selected.shape[1])
                            results["Way_of_Selection"].append("full")
                            results["Classifiers"].append(f"{self.name}")
                        else:
                            if (self.config_rncv["sfm"]) and ((estimator == "RandomForestClassifier") or 
                                (estimator == "XGBClassifier") or 
                                (estimator == 'GradientBoostingClassifier') or 
                                (estimator == "LGBMClassifier") or 
                                (estimator == "CatBoostClassifier")):
                                fs_type = "sfm"
                            else:
                                fs_type = self.config_rncv["feature_selection_type"]
                            results["Classifiers"].append(
                                f"{self.name}_{fs_type}_{num_feature}"
                            )
                            results["Selected_Features"].append(
                                X_train_selected.columns.tolist()
                            )
                            results["Number_of_Features"].append(num_feature)
                            results["Way_of_Selection"].append(
                                fs_type
                            )

                        # Track predictions
                        samples_counts = np.zeros(len(self.y))
                        for idx, resu, pred in zip(test_index, y_test, y_pred):
                            if pred == resu:
                                samples_counts[idx] += 1

                        results['Samples_counts'].append(samples_counts)
                        time.sleep(0.5)
                        
        # Check for consistent list lengths
        lengths = {key: len(val) for key, val in results.items()}

        if len(set(lengths.values())) > 1:
            print("Inconsistent lengths in results:", lengths)
            raise ValueError("Inconsistent lengths in results dictionary")

        return [results]

    def _outer_loop(self, i, avail_thr):
        start = time.time()  # Count time of outer loops

        # Split the data into train and test
        self.config_rncv["inner_cv"] = StratifiedKFold(
            n_splits=self.config_rncv["inner_splits"], shuffle=True, random_state=i
        )
        self.config_rncv["outer_cv"] = StratifiedKFold(
            n_splits=self.config_rncv["outer_splits"], shuffle=True, random_state=i
        )

        train_test_indices = list(self.config_rncv["outer_cv"].split(self.X, self.y))

        # Store the results in a list od dataframes
        list_dfs = []

        # Initiate the progress bar
        widgets = [
            progressbar.Percentage(),
            " ",
            progressbar.GranularBar(),
            " ",
            progressbar.Timer(),
            " ",
            progressbar.ETA(),
        ]
        
        # Find the parallelization method
        if self.config_rncv["parallel"] == "freely_parallel":
            temp_list = []
            with progressbar.ProgressBar(
                prefix=f"Outer fold of {i+1} round:",
                max_value=self.config_rncv["outer_splits"],
                widgets=widgets,
            ) as bar:
                split_index = 0

                # For each outer fold perform the inner loop
                for train_index, test_index in train_test_indices:
                    results = self._inner_loop(
                        train_index, test_index, self.X, self.y, avail_thr, i
                    )
                    temp_list.append(results)
                    bar.update(split_index)
                    split_index += 1
                    time.sleep(1)
                list_dfs = [item for sublist in temp_list for item in sublist]
                end = time.time()
        else:
            temp_list = []
            with progressbar.ProgressBar(
                prefix=f"Outer fold of {i+1} round:",
                max_value=self.config_rncv["outer_splits"],
                widgets=widgets,
            ) as bar:
                split_index = 0

                # For each outer fold perform the inner loop
                for train_index, test_index in train_test_indices:
                    results = self._inner_loop(
                        train_index, test_index, self.X, self.y, avail_thr, i
                    )
                    temp_list.append(results)
                    bar.update(split_index)
                    split_index += 1
                    time.sleep(1)
                list_dfs = [item for sublist in temp_list for item in sublist]
                end = time.time()

        # Return the list of dataframes and the time of the outer loop
        print(f"Finished with {i+1} round after {(end-start)/3600:.2f} hours.")
        return list_dfs

    # Function to insert new data into the updated database schema
    def _insert_data_into_db(self, scores_dataframe, config):
        try:
            credentials_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "db_credentials",
                "credentials.json"
            )
            
            with open(credentials_path, "r") as file:
                db_credentials = json.load(file)

            # Establish a connection to the PostgreSQL database
            connection = psycopg2.connect(
                dbname=db_credentials["db_name"],
                user=db_credentials["db_user"],
                password=db_credentials["db_password"],
                host=db_credentials["db_host"],
                port=db_credentials["db_port"]
            )
            cursor = connection.cursor()
            
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return
        
        try:
            # Insert dataset
            dataset_query = """
                INSERT INTO Datasets (dataset_name)
                VALUES (%s) ON CONFLICT (dataset_name) DO NOTHING RETURNING dataset_id;
            """
            cursor.execute(dataset_query, (config['dataset_name'],))
            dataset_id = cursor.fetchone()

            if dataset_id is None:
                cursor.execute("SELECT dataset_id FROM Datasets WHERE dataset_name = %s;", (config['dataset_name'],))
                dataset_id = cursor.fetchone()[0]
            else:
                dataset_id = dataset_id[0]
                
            # Insert job parameters
            if config['model_selection_type'] == 'rncv':
                job_parameters_query = """
                    INSERT INTO Job_Parameters (
                        n_trials, rounds, feature_selection_type, feature_selection_method, 
                        inner_scoring, outer_scoring, inner_splits, outer_splits, normalization, 
                        missing_values_method, class_balance
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING job_id;
                """
                cursor.execute(
                    job_parameters_query,
                    (
                        config['n_trials'], config['rounds'], config['feature_selection_type'],
                        config['feature_selection_method'], config['inner_scoring'], config['outer_scoring'],
                        config['inner_splits'], config['outer_splits'], config['normalization'],
                        config['missing_values_method'], config['class_balance']
                    )
                )
            else: 
                job_parameters_query = """
                    INSERT INTO Job_Parameters (
                        rounds, feature_selection_type, feature_selection_method, 
                        outer_scoring, outer_splits, normalization, 
                        missing_values_method, class_balance
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING job_id;
                """
                cursor.execute(
                    job_parameters_query,
                    (
                        config['rounds'], config['feature_selection_type'],
                        config['feature_selection_method'],config['scoring'],
                        config['splits'], config['normalization'],
                        config['missing_values_method'], config['class_balance']
                    )
                )
                
            job_id = cursor.fetchone()[0]

            # Insert classifiers and associated data
            for _, row in scores_dataframe.iterrows():
                # Check if the classifier combination already exists
                check_query = """
                    SELECT classifier_id FROM Classifiers
                    WHERE estimator = %s AND inner_selection = %s;
                """
                cursor.execute(check_query, (row["Est"], row["In_sel"]))
                classifier_id = cursor.fetchone()

                if classifier_id:
                    classifier_id = classifier_id[0]
                else:
                    classifier_query = """
                        INSERT INTO Classifiers (estimator, inner_selection)
                        VALUES (%s, %s) RETURNING classifier_id;
                    """
                    cursor.execute(classifier_query, (row["Est"], row["In_sel"]))
                    classifier_id = cursor.fetchone()[0]

                # Insert hyperparameters
                hyperparameters_query = """
                    INSERT INTO Hyperparameters (hyperparameters)
                    VALUES (%s) RETURNING hyperparameter_id;
                """
                hyperparameters = row["Hyp"]
                if isinstance(hyperparameters, np.ndarray):
                    hyperparameters = [dict(item) for item in hyperparameters]
                cursor.execute(hyperparameters_query, (json.dumps(hyperparameters),))
                hyperparameter_id = cursor.fetchone()[0]

                # Insert feature selection data
                feature_selection_query = """
                    INSERT INTO Feature_Selection (way_of_selection, numbers_of_features)
                    VALUES (%s, %s) RETURNING selection_id;
                """
                cursor.execute(feature_selection_query, (row["Sel_way"], row["Fs_num"]))
                selection_id = cursor.fetchone()[0]

                # Insert performance metrics
                performance_metrics_query = """
                    INSERT INTO Performance_Metrics (matthews_corrcoef, roc_auc, accuracy, balanced_accuracy, recall, precision, f1, specificity, average_precision)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING performance_id;
                """
                metrics = [
                    json.dumps(metric.tolist() if isinstance(metric, np.ndarray) else metric)
                    for metric in [row.get(metric) for metric in config['extra_metrics']]
                ]
                cursor.execute(performance_metrics_query, metrics)
                performance_id = cursor.fetchone()[0]

                # Insert samples classification rates
                samples_classification_query = """
                    INSERT INTO Samples_Classification_Rates (samples_classification_rates)
                    VALUES (%s) RETURNING sample_rate_id;
                """
                cursor.execute(samples_classification_query, (json.dumps(row["Classif_rates"]),))
                sample_rate_id = cursor.fetchone()[0]

                # Insert data into job combinations
                job_combinations_query = """
                    INSERT INTO Job_Combinations (
                        job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, model_selection_type
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING combination_id;
                """
                cursor.execute(
                    job_combinations_query,
                    (job_id, classifier_id, dataset_id, selection_id, hyperparameter_id, performance_id, sample_rate_id, config['model_selection_type'])
                )
                combination_id = cursor.fetchone()[0]

                # Insert feature counts
                if row["Sel_feat"] is not None:
                    selected_features = row["Sel_feat"]
                    if isinstance(selected_features, np.ndarray):
                        selected_features = selected_features.tolist()

                    if any(isinstance(i, list) for i in selected_features):
                        selected_features = [item for sublist in selected_features for item in sublist]

                    # Count occurrences of each feature
                    feature_counts = Counter([feature for feature in selected_features if feature])
                    feature_counts_query = """
                        INSERT INTO Feature_Counts (feature_name, count, combination_id)
                        VALUES %s;
                    """
                    
                    # Prepare feature values with combination_id included
                    feature_values = [(feat, count, combination_id) for feat, count in feature_counts.items()]
                    execute_values(cursor, feature_counts_query, feature_values)

            # Commit the transaction
            connection.commit()
            print("Data inserted into the database successfully.")
        except Exception as e:
            connection.rollback()
            print(f"An error occurred while inserting data into the database: {e}")
        finally:
            cursor.close()
            connection.close()
            
    def _file_name(self,config):
        default_values = {
            "rounds": 10,
            "n_trials": 100,
            "feature_selection_type": "mrmr",
            "feature_selection_method": "chi2",
            "inner_scoring": "matthews_corrcoef",
            "outer_scoring": "matthews_corrcoef",
            "inner_splits": 5,
            "outer_splits": 5,
            "normalization": "minmax",
            "class_balance": "auto",    
            "sfm": False,
            "missing_values": "median",
            "num_features": None,
            "scoring": "matthews_corrcoef",
            "splits": 5
            
        }
        name_add = ""
        for conf in config:
            if conf in default_values.keys():
                if config[conf] != default_values[conf]:
                    name_add += f"_{conf}_{config[conf]}"
        name_add += f"_{datetime.now().strftime('%Y%m%d_%H%M')}"
        return name_add
     
    def _return_csv(self, final_dataset_name, scores_dataframe, extra_metrics=None, filter_csv=None):
        results_path = f"{final_dataset_name}_outerloops_results.csv"
        cols_drop = ["Classif_rates", "Clf", "Hyp", "Sel_feat"]
        if extra_metrics is not None:
            for metric in extra_metrics:
                cols_drop.append(f"{metric}") 
        statistics_dataframe = scores_dataframe.drop(cols_drop, axis=1)
        if filter_csv != None:
            try:
                for mtrc_stat in filter_csv:
                    if 'h' in filter_csv[mtrc_stat]:
                        statistics_dataframe = statistics_dataframe[statistics_dataframe[mtrc_stat] >= filter_csv[mtrc_stat]['h']]
                    elif 'l' in filter_csv[mtrc_stat]:
                        statistics_dataframe = statistics_dataframe[statistics_dataframe[mtrc_stat] <= filter_csv[mtrc_stat]['l']]
            except Exception as e:
                print(f'An error occurred while filtering the final csv file: {e}\nThe final csv file will not be filtered.')
        statistics_dataframe.to_csv(results_path, index=False)
        print(f"Statistics results saved to {results_path}")
        return statistics_dataframe
    
    def _input_renamed_metrics(self, extra_metrics, results, indices):
        # Metrics dict
        metric_abbreviations = {
            'roc_auc': 'AUC',
            'accuracy': 'ACC',
            'balanced_accuracy': 'BAL_ACC',
            'recall': 'REC',
            'precision': 'PREC',
            'f1': 'F1',
            'average_precision': 'AVG_PREC',
            'specificity': 'SPEC',
            'matthews_corrcoef': 'MCC'
        }
        
        # Add metrics
        for metric in extra_metrics:
            qck_mtrc = metric_abbreviations[f"{metric}"]
            metric_values = indices[f"{metric}"].values

            results[-1][f"{metric}"] = metric_values  # If this stores an array, keep it as-is

            # Round metrics to 3 decimal places
            results[-1][f"{qck_mtrc}_mean"] = round(np.mean(metric_values), 3)
            results[-1][f"{qck_mtrc}_std"] = round(np.std(metric_values), 3)
            results[-1][f"{qck_mtrc}_sem"] = round(sem(metric_values), 3)
            # Compute lower and upper percentage values and round to 3 decimal places
            lower_percentile = np.percentile(metric_values, 5)
            upper_percentile = np.percentile(metric_values, 95)
            results[-1][f"{qck_mtrc}_lowerCI"] = round(lower_percentile, 3)
            results[-1][f"{qck_mtrc}_upperCI"] = round(upper_percentile, 3)
            results[-1][f"{qck_mtrc}_med"] = round(np.median(metric_values), 3)
            # Bootstrap confidence intervals for median and mean
            lomed, upmed = self._bootstrap_ci(metric_values, type='median')
            lomean, upmean = self._bootstrap_ci(metric_values, type='mean')
            results[-1][f"{qck_mtrc}_lomean"] = round(lomean, 3)
            results[-1][f"{qck_mtrc}_upmean"] = round(upmean, 3)
            results[-1][f"{qck_mtrc}_lomed"] = round(lomed, 3)
            results[-1][f"{qck_mtrc}_upmed"] = round(upmed, 3)
        
        return results
            
    def nested_cv(
        self,
        n_trials: int = 100,
        rounds: int = 10,
        exclude: str|list = None,
        search_on: str|list = None,
        info_to_db: bool = False,
        num_features: int|list = None,
        feature_selection_type: str = "mrmr",
        feature_selection_method: str = "chi2",
        sfm: bool = False,
        freq_feat: int = None,
        class_balance: str = 'auto',
        inner_scoring: str = "matthews_corrcoef",
        outer_scoring: str = "matthews_corrcoef",
        inner_selection_lst: list = ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"],
        extra_metrics: str|list = ['recall', 'specificity', 'accuracy', 'balanced_accuracy', 
                            'precision', 'f1', 'roc_auc', 'average_precision', 'matthews_corrcoef'],
        plot: str = "box",
        inner_splits: int = 5,
        outer_splits: int = 5,
        parallel: str = "thread_per_round",
        normalization: str = "minmax",
        missing_values_method: str = "median",
        return_csv: bool = True,
        filter_csv: dict = None
    ):
        """
        Perform model selection using Nested Cross Validation and visualize the selected features' frequency.
        """

        # Missing values manipulation
        if missing_values_method == "drop":
            print(
                "Values cannot be dropped at ncv because of inconsistent shapes. \nThe missing values with automaticly replaced by the median of each feature."
            )
            missing_values_method = "median"
        if self.X.isnull().values.any():
            print(
                f"Your Dataset contains NaN values. Some estimators does not work with NaN values.\nThe {missing_values_method} method will be used for the missing values manipulation.\n"
            )
            
        if extra_metrics is not None:
            if type(extra_metrics) is not list:
                extra_metrics = [extra_metrics]
            for metric in extra_metrics:
                scoring_check(metric)
            print('All the extra metrics are valid.')
            if outer_scoring not in extra_metrics:
                extra_metrics.insert(0, outer_scoring)
            elif outer_scoring in extra_metrics and extra_metrics.index(outer_scoring) != 0:
                # Remove it from its current position
                extra_metrics.remove(outer_scoring)
                # Insert it at the first index
                extra_metrics.insert(0, outer_scoring)
        else:
            extra_metrics = [outer_scoring]
        if class_balance not in ['auto','smote','smote_enn','adasyn','borderline_smote','tomek', None]:
            raise ValueError("class_balance must be one of the following: 'auto','smote','smotenn','adasyn','borderline_smote','tomek', or None")
        elif class_balance == None:
            class_balance = 'auto'
            print('Class balance is set to "auto"')
            
        # Set parameters for the nested functions of the ncv process
        self.config_rncv = locals()
        self.config_rncv.pop("self", None)
        self.config_rncv['dataset_name'] = self.csv_dir
        self.config_rncv['model_selection_type'] = 'rncv'
        
        # Set available classifiers
        if search_on is not None:
            classes = search_on  # 'search_on' is a list of classifier names as strings
            exclude_classes = [
                clf for clf in self.available_clfs.keys() if clf not in classes
            ]
        elif exclude is not None:
             exclude_classes = (
                exclude  # 'exclude' is a list of classifier names as strings
            )
        else:
            exclude_classes = []

        # Filter classifiers based on the exclude_classes list
        clfs = [clf for clf in self.available_clfs.keys() if clf not in exclude_classes]
        self.config_rncv["clfs"] = clfs

        # Checks for reliability of parameters
        if (inner_scoring not in sklearn.metrics.get_scorer_names()) and (inner_scoring != "specificity"):
            raise ValueError(
                f"Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
            )
        if (outer_scoring not in sklearn.metrics.get_scorer_names()) and (outer_scoring != "specificity"):
            raise ValueError(
                f"Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
            )
        for inner_selection in inner_selection_lst:
            if inner_selection not in ["validation_score", "one_sem", "gso_1", "gso_2","one_sem_grd"]:
                raise ValueError(
                    f'Invalid inner method: {inner_selection}. Select one of the following: ["validation_score", "one_sem", "one_sem_grd", "gso_1", "gso_2"]'
                )

        # Parallelization
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        if num_cores < rounds:
            use_cores = num_cores
        else:
            use_cores = rounds
        avail_thr = max(1, num_cores // rounds)

        if parallel == "thread_per_round":
            avail_thr = 1
            with threadpool_limits(limits=avail_thr):
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(self._outer_loop)(i, avail_thr) for i in trial_indices
                )
        elif parallel == "freely_parallel":
            with threadpool_limits():
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(self._outer_loop)(i, avail_thr) for i in trial_indices
                )
        else:
            raise ValueError(
                f"Invalid parallel option: {parallel}. Select one of the following: thread_per_round or freely_parallel"
            )

        list_dfs_flat = list(chain.from_iterable(list_dfs))
        
        # Create results dataframe
        results = []
        df = pd.DataFrame()
        for item in list_dfs_flat:
            dataframe = pd.DataFrame(item)
            df = pd.concat([df, dataframe], axis=0)

        for inner_selection in inner_selection_lst:
            df_inner = df[df["Inner_selection_mthd"] == inner_selection]
            for classif in np.unique(df_inner["Classifiers"]):
                indices = df_inner[df_inner["Classifiers"] == classif]
                filtered_scores = indices[f"{self.config_rncv['outer_scoring']}"].values
                if num_features is not None:
                    filtered_features = indices["Selected_Features"].values
                mean_score = np.mean(filtered_scores)
                max_score = np.max(filtered_scores)
                std_score = np.std(filtered_scores)
                sem_score = sem(filtered_scores)
                median_score = np.median(filtered_scores)
                Numbers_of_Features = indices["Number_of_Features"].unique()[0]
                Way_of_Selection = indices["Way_of_Selection"].unique()[0]
                samples_classification_rates = np.zeros(len(self.y))
                for test_part in indices["Samples_counts"]:
                    samples_classification_rates=np.add(samples_classification_rates,test_part)
                samples_classification_rates /= rounds
                                
                results.append(
                    {
                        "Est": df_inner[df_inner["Classifiers"] == classif]["Estimator"].unique()[
                            0
                        ],
                        "Clf": classif,
                        "Hyp": df_inner[df_inner["Classifiers"] == classif][
                            "Hyperparameters"
                        ].values,
                        "Sel_way": Way_of_Selection,
                        "Fs_inner": feature_selection_method,
                        "Fs_num": Numbers_of_Features,
                        "Sel_feat": filtered_features
                        if num_features is not None
                        else None,
                        "Norm": normalization,
                        "Miss_vals": missing_values_method,
                        "In_cv": inner_splits,
                        "Out_cv": outer_splits,
                        "Rnds": rounds,
                        "Trials": n_trials,
                        "Class_blnc": class_balance,
                        "In_scor": inner_scoring,
                        "Out_scor": outer_scoring,
                        "In_sel":inner_selection,
                        "Classif_rates": samples_classification_rates.tolist(),
                    }
                )

                results = self._input_renamed_metrics(
                    extra_metrics, results, indices
                )
                                            
        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)
        
        # Create a 'Results' directory
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Name
        final_dataset_name = self._name_outputs(self.config_rncv, results_dir)  
            
        # Save the results to a CSV file of the outer scores for each classifier
        if return_csv:
            statistics_dataframe = self._return_csv(final_dataset_name, scores_dataframe, extra_metrics, filter_csv)
            
        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:    
            # Plot histogram of features
            self._histogram(scores_dataframe, final_dataset_name, freq_feat, clfs)

        
        # Plot box or violin plots of the outer cross-validation scores for all Inner_Selection methods
        if plot is not None:
            self._plot(scores_dataframe, plot, self.config_rncv['outer_scoring'], final_dataset_name)
            
        if info_to_db:
            # Add to database
            self._insert_data_into_db(scores_dataframe, self.config_rncv)
            
        return statistics_dataframe