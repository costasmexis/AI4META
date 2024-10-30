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
                    f"Outer_{self.config_rcv['scoring']}": [],
                    "Classifiers": [],
                    "Selected_Features": [],
                    "Number_of_Features": [],
                    "Way_of_Selection": [],
                    "Estimator": [],
                    'Samples_counts': [],
                }
                if self.config_rcv["extra_metrics"] != None:
                    results.update({f"{metric}": [] for metric in self.config_rcv["extra_metrics"]})

                # Fold over the number of features
                for num_feature2_use in feature_loop:
                    X_train_selected, X_test_selected, num_feature = self._filter_features(
                        train_index, test_index, self.X, self.y, num_feature2_use, cvncvsel='rcv'
                    )
                    y_train, y_test = self.y[train_index], self.y[test_index]

                    # Check of the classifiers given list
                    if self.config_rcv["clfs"] is None:
                        raise ValueError("No classifier specified.")
                    else:
                        for estimator in self.config_rcv["clfs"]:
                            # For every estimator find the best hyperparameteres
                            self.name = estimator
                            self.estimator = self.available_clfs[estimator]
                            clf = self._create_model_instance(
                                    self.name, params=None
                                )
                            clf.fit(X_train_selected, y_train)
                            
                            # Store the results and apply one_sem method if its selected
                            results["Estimator"].append(self.name)
                            results[f"Outer_{self.config_rcv['scoring']}"].append(
                                clf.score(X_test_selected, y_test)
                            )
                            if self.config_rcv["extra_metrics"] != None:
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
                                    f"{self.name}_{self.config_rncv['feature_selection_type']}_{num_feature}"
                                )
                                results["Selected_Features"].append(
                                    X_train_selected.columns.tolist()
                                )
                                results["Number_of_Features"].append(num_feature)
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
        normalization="minmax",
        missing_values_method="median",
        name_add=None,
        extra_metrics=['roc_auc','accuracy','balanced_accuracy','recall','precision','f1'],
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

        # Set parameters for the nested functions of the cv process
        self.config_rcv = locals()
        self.config_rcv.pop("self", None)

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

        # # Checks for reliability of parameters
        # if (scoring not in sklearn.metrics.get_scorer_names()) or (scoring is not 'specificity'):
        #     raise ValueError(
        #         f"Invalid outer scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
        #     )
        
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
            filtered_scores = indices[f"Outer_{self.config_rcv['scoring']}"].values
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
                    "Estimator": df[df["Classifiers"] == classif]["Estimator"].unique()[
                        0
                    ],
                    "Classifier": classif,
                    f"Outer_{self.config_rcv['scoring']}": filtered_scores.tolist(),
                    "Max": max_score,
                    "Std": std_score,
                    "SEM": sem_score,
                    "Median": median_score,
                    "Hyperparameters": "Default",
                    "Selected_Features": filtered_features
                    if num_features is not None
                    else None,
                    "Numbers_of_Features": Numbers_of_Features,
                    "Way_of_Selection": Way_of_Selection,
                    "Samples_classification_rates": samples_classification_rates.tolist(),
                }
            )

            # Add extra metrics
            if extra_metrics is not None:
                for metric in extra_metrics:
                    results[-1][f"{metric}"] = indices[f"{metric}"].values

        print(f"Finished with {len(results)} estimators")
        scores_dataframe = pd.DataFrame(results)

        # Create a 'Results' directory
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Initiate name
        if num_features is not None:
            features = num_features
        else:
            features = "all_features"

        try:
            dataset_name = self._set_result_csv_name(self.csv_dir)
            if name_add is None:
                results_name = f"{dataset_name}_{'validation_score'}_{features}_RCV"
            else:
                results_name = f"{dataset_name}_{'validation_score'}_{features}_{name_add}_RCV"
            final_dataset_name = os.path.join(results_dir, results_name)
        except Exception as e:
            dataset_name = "dataset"
            if name_add is None:
                results_name = f"{dataset_name}_{'validation_score'}_{features}_RCV"
            else:
                results_name = f"{dataset_name}_{'validation_score'}_{features}_{name_add}_RCV"
            final_dataset_name = os.path.join(
                results_dir, results_name)

        # Save the results to a CSV file of the outer scores for each classifier
        if return_csv:
            results_path = f"{final_dataset_name}_outerloops_results.csv"
            scores_dataframe.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")

        # Plot box or violin plots of the outer cross-validation scores
        if plot is not None:
            scores_long = scores_dataframe.explode(f"Outer_{self.config_rcv['scoring']}")
            scores_long[f"Outer_{self.config_rcv['scoring']}"] = scores_long[f"Outer_{self.config_rcv['scoring']}"].astype(float)

            fig = go.Figure()
            if plot == "box":
                # Add box plots for each classifier
                for classifier in scores_dataframe["Classifier"]:
                    data = scores_long[scores_long["Classifier"] == classifier][
                        f"Outer_{self.config_rcv['scoring']}"
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

            elif plot == "violin":
                for classifier in scores_dataframe["Classifier"]:
                    data = scores_long[scores_long["Classifier"] == classifier][
                        f"Outer_{self.config_rcv['scoring']}"
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
                title="Model Selection Results",
                yaxis_title=f"Scores {self.config_rcv['scoring']}",
                xaxis_title="Classifier",
                xaxis_tickangle=-45,
                template="plotly_white",
            )
            # Save the figure as an image in the "Results" directory
            image_path = f"{final_dataset_name}_model_selection_plot.png"
            fig.write_image(image_path)
            fig.show()
        else:
            pass

        return scores_dataframe

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

    def _inner_loop(self, train_index, test_index, X, y, avail_thr):
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
            f"Outer_{self.config_rncv['inner_scoring']}": [],
            "Classifiers": [],
            "Selected_Features": [],
            "Number_of_Features": [],
            "Hyperparameters": [],
            "Way_of_Selection": [],
            "Estimator": [],
            'Samples_counts': [],
            'Inner_selection_mthd': [],
        }
        if self.config_rncv["extra_metrics"] != None:
            results.update({f"{metric}": [] for metric in self.config_rncv["extra_metrics"]})

        # Loop over the number of features
        for num_feature2_use in feature_loop:
            if not self.config_rncv['sfm']:
                X_train_selected, X_test_selected, num_feature = self._filter_features(
                    train_index, test_index, X, y, num_feature2_use, cvncvsel='rncv'
                )
                y_train, y_test = y[train_index], y[test_index]

                
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
                            train_index, test_index, X, y, num_feature2_use=X.shape[1],cvncvsel='rncv'
                        )
                        y_train, y_test = y[train_index], y[test_index]

                        # Assuming estimator is already instantiated and fitted
                        X_train_selected, X_test_selected, num_feature = self._sfm(self.estimator, X_train_selected, X_test_selected, y_train, num_feature2_use)   
                    elif self.config_rncv["sfm"]:
                        continue
                    else:pass
                    
                    self._set_optuna_verbosity(logging.ERROR)
                    clf = optuna.integration.OptunaSearchCV(
                        estimator=self.estimator,
                        scoring=self.config_rncv["inner_scoring"],
                        param_distributions=optuna_grid[opt_grid][self.name],
                        cv=self.config_rncv["inner_cv"],
                        return_train_score=True,
                        n_jobs=n_jobs,
                        verbose=0,
                        n_trials=self.config_rncv["n_trials_ncv"],
                    )
                    clf.fit(X_train_selected, y_train)
           
                    for inner_selection in self.config_rncv["inner_selection_lst"]:
                        results['Inner_selection_mthd'].append(inner_selection)
                        # Store the results and apply one_sem method if its selected
                        results["Estimator"].append(self.name)
                        if inner_selection == "validation_score":
                            results[f"Outer_{self.config_rncv['outer_scoring']}"].append(
                                get_scorer(self.config_rncv["outer_scoring"])(
                                    clf, X_test_selected, y_test
                                )
                            )
                            results["Hyperparameters"].append(clf.best_params_)
                            if self.config_rncv["extra_metrics"] != None:
                                for metric in self.config_rncv["extra_metrics"]:
                                    if metric == 'specificity':
                                        results[f"{metric}"].append(
                                            self._specificity_scorer(clf, X_test_selected, y_test)
                                        )
                                    else:
                                        results[f"{metric}"].append(
                                            get_scorer(metric)(clf, X_test_selected, y_test)
                                        )
                            y_pred = clf.predict(X_test_selected)
                        else:
                            trials = clf.trials_
                            if (inner_selection == "one_sem") or (inner_selection == "one_sem_grd"):
                                samples = X_train_selected.shape[0]
                                # Find simpler parameters with the one_sem method if there are any
                                simple_model_params = self._one_sem_model(trials, self.name, samples, self.config_rncv['inner_splits'],inner_selection)
                            elif (inner_selection == "gso_1") or (inner_selection == "gso_2"):
                                # Find parameters with the smaller gap score with gso_1 method if there are any
                                simple_model_params = self._gso_model(trials, self.name, self.config_rncv['inner_splits'],inner_selection)
                            results["Hyperparameters"].append(simple_model_params)
                            # Fit the new model
                            new_params_clf = self._create_model_instance(
                                self.name, simple_model_params
                            )
                            new_params_clf.fit(X_train_selected, y_train)
                            results[f"Outer_{self.config_rncv['outer_scoring']}"].append(
                                new_params_clf.score(X_test_selected, y_test)
                            )
                            
                            if self.config_rncv["extra_metrics"] != None:
                                for metric in self.config_rncv["extra_metrics"]:
                                    if metric == 'specificity':
                                        results[f"{metric}"].append(
                                            self._specificity_scorer(new_params_clf, X_test_selected, y_test)
                                        )
                                    else:
                                        results[f"{metric}"].append(
                                            get_scorer(metric)(new_params_clf, X_test_selected, y_test)
                                        )
                            y_pred = new_params_clf.predict(X_test_selected)

                        # Store the results using different names if feature selection is applied
                        if num_feature == "full" or num_feature is None:
                            results["Selected_Features"].append(None)
                            results["Number_of_Features"].append(X_test_selected.shape[1])
                            results["Way_of_Selection"].append("full")
                            results["Classifiers"].append(f"{self.name}")
                        else:
                            results["Classifiers"].append(
                                f"{self.name}_{self.config_rncv['feature_selection_type']}_{num_feature}"
                            )
                            results["Selected_Features"].append(
                                X_train_selected.columns.tolist()
                            )
                            results["Number_of_Features"].append(num_feature)
                            results["Way_of_Selection"].append(
                                self.config_rncv["feature_selection_type"]
                            )

                        # Track predictions
                        samples_counts = np.zeros(len(self.y))
                        for idx, resu, pred in zip(test_index, y_test, y_pred):
                            if pred == resu:
                                samples_counts[idx] += 1

                        results['Samples_counts'].append(samples_counts)
                        time.sleep(0.5)
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
                        train_index, test_index, self.X, self.y, avail_thr
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
                        train_index, test_index, self.X, self.y, avail_thr
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

    def nested_cv(
        self,
        n_trials_ncv=100,
        rounds=10,
        exclude=None,
        freq_feat=None,
        search_on=None,
        num_features=None,
        feature_selection_type="mrmr",
        return_csv=True,
        feature_selection_method="chi2",
        plot="box",
        inner_scoring="matthews_corrcoef",
        inner_selection_lst=["validation_score","one_sem","gso_1","gso_2","one_sem_grd"],
        outer_scoring="matthews_corrcoef",
        inner_splits=5,
        outer_splits=5,
        normalization="minmax",
        parallel="thread_per_round",
        missing_values_method="median",
        frfs=None,
        name_add=None,
        extra_metrics=['roc_auc','accuracy','balanced_accuracy','recall','precision','f1', 'average_precision','specificity'],
        show_bad_samples=False,
        sfm=False,
        info_to_db = False
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

        # Set parameters for the nested functions of the ncv process
        self.config_rncv = locals()
        self.config_rncv.pop("self", None)

        if num_features is not None:
            print(
                f"The num_features parameter is {num_features}.\nThe result will be a Dataframe and a List with the freq_feat number of the most important features.\nIf the freq_feat is None, the result will be a List with all features."
            )
        if (frfs is not None) and (num_features is None):
            print(
                "You are using the frfs parameter and not the num_features. The results will not contain the most important features."
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
        self.config_rncv["clfs"] = clfs

        # Checks for reliability of parameters
        if inner_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid inner scoring metric: {inner_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
            )
        if outer_scoring not in sklearn.metrics.get_scorer_names():
            raise ValueError(
                f"Invalid outer scoring metric: {outer_scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
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
                filtered_scores = indices[f"Outer_{self.config_rncv['outer_scoring']}"].values
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
                        "Estimator": df_inner[df_inner["Classifiers"] == classif]["Estimator"].unique()[
                            0
                        ],
                        "Classifier": classif,
                        f"{self.config_rncv['outer_scoring']}": filtered_scores.tolist(),
                        f"Max_{self.config_rncv['outer_scoring']}": max_score,
                        f"Std_{self.config_rncv['outer_scoring']}": std_score,
                        f"SEM_{self.config_rncv['outer_scoring']}": sem_score,
                        f"Median_{self.config_rncv['outer_scoring']}": median_score,
                        "Hyperparameters": df_inner[df_inner["Classifiers"] == classif][
                            "Hyperparameters"
                        ].values,
                        "Selected_Features": filtered_features
                        if num_features is not None
                        else None,
                        "Inner_Selection":inner_selection,
                        "Numbers_of_Features": Numbers_of_Features,
                        "Way_of_Selection": Way_of_Selection,
                        "Samples_classification_rates": samples_classification_rates.tolist(),
                    }
                )

                # Add extra metrics
                if extra_metrics is not None:
                    for metric in extra_metrics:
                        results[-1][f"{metric}"] = indices[f"{metric}"].values
                        results[-1][f"Max_{metric}"] = np.max(indices[f"{metric}"].values)
                        results[-1][f"Std_{metric}"] = np.std(indices[f"{metric}"].values)
                        results[-1][f"SEM_{metric}"] = sem(indices[f"{metric}"].values)
                        results[-1][f"Median_{metric}"] = np.median(indices[f"{metric}"].values)
                        
        print(f"Finished with {len(results)} estimators")
        scores_dataframe = pd.DataFrame(results)
        
        # Create a 'Results' directory
        results_dir = "Results_ncv"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Initiate name
        if num_features is not None:
            features = num_features
        else:
            features = "all_features"

        try:
            dataset_name = self._set_result_csv_name(self.csv_dir)
            if name_add is None:
                results_name = f"{dataset_name}_{inner_selection_lst}_{features}"
            else:
                results_name = f"{dataset_name}_{inner_selection_lst}_{features}_{name_add}"
            final_dataset_name = os.path.join(results_dir, results_name)
        except Exception as e:
            dataset_name = "dataset"
            if name_add is None:
                results_name = f"{dataset_name}_{inner_selection_lst}_{features}"
            else:
                results_name = f"{dataset_name}_{inner_selection_lst}_{features}_{name_add}"
            final_dataset_name = os.path.join(
                results_dir, f"{dataset_name}_{inner_selection_lst}_{features}"
            )
        
        if show_bad_samples:
            threshold = 0.5
            inner_selection_methods = scores_dataframe['Inner_Selection'].unique()
            limit = 0  # Counter to track how many Inner_Selection methods have no bad samples
            
            fig = go.Figure()
            
            for inner_selection_method in inner_selection_methods:
                df_inner_selection = scores_dataframe[scores_dataframe['Inner_Selection'] == inner_selection_method]
                classifiers = df_inner_selection['Classifier'].unique()
                
                for classifier in classifiers:
                    df_classifier = df_inner_selection[df_inner_selection['Classifier'] == classifier]
                    samples_classification_rates = np.array(df_classifier['Samples_classification_rates'].values[0])
                    
                    # Find bad samples (classification rate < threshold)
                    bad_samples_indices = np.where(samples_classification_rates < threshold)[0]
                    
                    # Skip plotting if no bad samples are found
                    if bad_samples_indices.size == 0:
                        print(f'No bad samples for {classifier} ({inner_selection_method}).')
                        limit += 1  # Increment limit if no bad samples are found
                        continue
                    
                    bad_samples_rates = samples_classification_rates[bad_samples_indices]              
                    
                    # Add the bar trace for bad samples (with wider bars)
                    fig.add_trace(go.Bar(
                        x=bad_samples_indices.tolist(),  # Ensure it's converted to a list for Plotly
                        y=bad_samples_rates.tolist(),    # Convert rates to list for Plotly
                        name=f'Bad Samples for {classifier} ({inner_selection_method})',
                        text=[f'{rate:.2f}' for rate in bad_samples_rates],
                        textposition='auto',
                        width=[0.5] * len(bad_samples_indices)  # Make bars wider (adjust the value for width)
                    ))
                    
                    # Add a scatter trace to mark points where the rate is 0
                    zero_rate_indices = bad_samples_indices[bad_samples_rates == 0]
                    if len(zero_rate_indices) > 0:
                        fig.add_trace(go.Scatter(
                            x=zero_rate_indices.tolist(),
                            y=[0] * len(zero_rate_indices),  # Place the points on y = 0
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='circle-open'),  # Customize marker appearance
                            name=f'Zero Rate Points for {classifier} ({inner_selection_method})',
                            showlegend=False  # Optionally, remove from legend if not needed
                        ))

            # Check if limit equals the number of inner_selection_methods before showing the plot
            if limit == len(inner_selection_methods):
                print('No bad samples found for any Inner_Selection method.')
            elif fig.data:  # Only show the plot if there are traces to plot
                fig.update_layout(
                    title='Bad Samples Classification Rates for Each Classifier and Inner Selection Method',
                    xaxis_title='Sample Index',
                    yaxis_title='Classification Rate',
                    legend_title='Classifiers (Inner_Selection)',
                    barmode='group',
                    width=self.X.shape[0] * 17,  
                    height=700
                )
                
                # fig.show()
                
                # Save the plot to 'Results/bad_samples.png'
                save_path = f"{final_dataset_name}_bad_samples.png"
                fig.write_image(save_path)
            else:
                print("No bad samples to plot.")

        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:
            if freq_feat == None:
                freq_feat = self.X.shape[1]
            elif freq_feat > self.X.shape[1]:
                freq_feat = self.X.shape[1]

            # Plot histogram of features
            feature_counts = Counter()
            for idx, row in scores_dataframe.iterrows():
                if row["Way_of_Selection"] != "full":  # If no features were selected, skip
                    features = list(
                        chain.from_iterable(
                            [list(index_obj) for index_obj in row["Selected_Features"]]
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

                # # Show the interactive plot
                # fig.show()

                # Save the plot to 'Results/histogram.png'
                save_path = f"{final_dataset_name}_histogram.png"
                fig.write_image(save_path)

            # Save the number of features that were most frequently selected
            features_list = [x[0] for x in sorted_features_counts]
            if (frfs is not None) and (frfs < len(features_list)):
                features_list = features_list[:frfs]
                print(f"Top {frfs} most frequently selected features: {features_list}")
            
        def bootstrap_median_ci(data, num_iterations=1000, ci=0.95):
            medians = []
            for _ in range(num_iterations):
                sample = np.random.choice(data, size=len(data), replace=True)
                medians.append(np.median(sample))
            lower_bound = np.percentile(medians, (1 - ci) / 2 * 100)
            upper_bound = np.percentile(medians, (1 + ci) / 2 * 100)
            return lower_bound, upper_bound

        # Plot box or violin plots of the outer cross-validation scores for all Inner_Selection methods
        if plot is not None:
            scores_long = scores_dataframe.explode(f"{self.config_rncv['outer_scoring']}")
            scores_long[f"{self.config_rncv['outer_scoring']}"] = scores_long[f"{self.config_rncv['outer_scoring']}"].astype(float)

            fig = go.Figure()

            inner_selection_methods = scores_dataframe['Inner_Selection'].unique()

            for inner_selection_method in inner_selection_methods:
                df_inner_selection = scores_long[scores_long['Inner_Selection'] == inner_selection_method]
                classifiers = df_inner_selection["Classifier"].unique()

                if plot == "box":
                    # Add box plots for each classifier within each Inner_Selection method
                    for classifier in classifiers:
                        data = df_inner_selection[df_inner_selection["Classifier"] == classifier][
                            f"{self.config_rncv['outer_scoring']}"
                        ]
                        median = np.median(data)
                        fig.add_trace(
                            go.Box(
                                y=data,
                                name=f"{classifier} ({inner_selection_method}) (Median: {median:.2f})",
                                boxpoints="all",
                                jitter=0.3,
                                pointpos=-1.8,
                            )
                        )

                        # Calculate and add 95% CI for the median
                        lower, upper = bootstrap_median_ci(data)
                        fig.add_trace(
                            go.Scatter(
                                x=[f"{classifier} ({inner_selection_method}) (Median: {median:.2f})",
                                f"{classifier} ({inner_selection_method}) (Median: {median:.2f})"],
                                y=[lower, upper],
                                mode="lines",
                                line=dict(color="black", dash="dash"),
                                showlegend=False,
                            )
                        )

                elif plot == "violin":
                    # Add violin plots for each classifier within each Inner_Selection method
                    for classifier in classifiers:
                        data = df_inner_selection[df_inner_selection["Classifier"] == classifier][
                            f"{self.config_rncv['outer_scoring']}"
                        ]
                        median = np.median(data)
                        fig.add_trace(
                            go.Violin(
                                y=data,
                                name=f"{classifier} ({inner_selection_method}) (Median: {median:.2f})",
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
                title="Model Selection Results by Classifier and Inner Selection Method",
                yaxis_title=f"Scores {self.config_rncv['outer_scoring']}",
                xaxis_title="Classifier (Inner Selection Method)",
                xaxis_tickangle=-45,
                template="plotly_white",
            )
            
            # Save the figure as an image in the "Results" directory
            image_path = f"{final_dataset_name}_model_selection_plot.png"
            fig.write_image(image_path)
            # fig.show()
        else:
            pass

        # Save the results to a CSV file of the outer scores for each classifier
        if return_csv:
            results_path = f"{final_dataset_name}_outerloops_results.csv"
            cols_drop = ["Samples_classification_rates", "Classifier",self.config_rncv['outer_scoring'],"Hyperparameters","Selected_Features"]
            for metric in extra_metrics:
                cols_drop.append(metric) 
            statistics_dataframe = scores_dataframe.drop(cols_drop, axis=1)
            statistics_dataframe.to_csv(results_path, index=False)
            print(f"Statistics results saved to {results_path}")
            
        if info_to_db:
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
                print("Connected to PostgreSQL database")
                
            except Exception as e:
                print(f"Error connecting to database: {e}")
                
            try:
                # Insert dataset
                dataset_query = """
                    INSERT INTO Datasets (dataset_name)
                    VALUES (%s) ON CONFLICT (dataset_name) DO NOTHING RETURNING dataset_id;
                """
                print(f"Dataset : {self.csv_dir}")

                # Execute the query with only the dataset name as a parameter
                cursor.execute(dataset_query, (self.csv_dir,))
                dataset_id = cursor.fetchone()
                
                # If dataset_id is None (i.e., dataset already exists), retrieve it
                if dataset_id is None:
                    cursor.execute("SELECT dataset_id FROM Datasets WHERE dataset_name = %s;", (self.csv_dir,))
                    dataset_id = cursor.fetchone()[0]
                else:
                    dataset_id = dataset_id[0]

                # Insert classifiers and associated data
                for _, row in scores_dataframe.iterrows():
                    classifier_query = """
                        INSERT INTO Classifiers (dataset_id, estimator, inner_selection)
                        VALUES (%s, %s, %s) RETURNING classifier_id;
                    """
                    print(f"Classifier : {dataset_id}, {row['Estimator']},  {row['Inner_Selection']}")
                    # Ensure the classifier row contains all required values
                    cursor.execute(classifier_query, (dataset_id, row["Estimator"], row["Inner_Selection"]))
                    classifier_id = cursor.fetchone()
                    
                    if classifier_id:
                        classifier_id = classifier_id[0]
                    else:
                        raise ValueError("Classifier ID could not be retrieved.")

                    # Insert hyperparameters
                    hyperparameters_query = """
                        INSERT INTO Hyperparameters (classifier_id, dataset_id, hyperparameters)
                        VALUES (%s, %s, %s)
                    """
                    # Ensure hyperparameters are JSON serializable
                    hyperparameters = row["Hyperparameters"]
                    if isinstance(hyperparameters, np.ndarray):
                        hyperparameters = [dict(item) for item in hyperparameters]
                    print(f"Hyperparameters : {hyperparameters}")
                    cursor.execute(hyperparameters_query, (classifier_id, dataset_id, json.dumps(hyperparameters)))

                    # Insert feature selection data
                    feature_selection_query = """
                        INSERT INTO Feature_Selection (classifier_id, dataset_id, way_of_selection, numbers_of_features)
                        VALUES (%s, %s, %s, %s) RETURNING selection_id;
                    """
                    # Debugging output for feature selection values
                    way_of_selection = row["Way_of_Selection"]
                    numbers_of_features = row["Numbers_of_Features"]
                    print(f"Selection : {classifier_id}, {dataset_id}, {way_of_selection}-{type(way_of_selection)}, {numbers_of_features}-{type(numbers_of_features)}")
                    cursor.execute(
                        feature_selection_query,
                        (classifier_id, dataset_id, way_of_selection, numbers_of_features)
                    )
                    selection_id = cursor.fetchone()[0]

                    # # Insert feature counts
                    # if num_features:
                    #     # Flatten the list if `Selected_Features` contains lists of lists
                    #     selected_features = row["Selected_Features"]
                    #     if isinstance(selected_features, list) and any(isinstance(i, list) for i in selected_features):
                    #         selected_features = [item for sublist in selected_features for item in sublist]

                    #     # Count occurrences of each feature
                    #     feature_counts = Counter(selected_features)
                    #     feature_counts_query = """
                    #         INSERT INTO Feature_Counts (feature_name, count, selection_id, dataset_id)
                    #         VALUES %s
                    #     """
                    #     print(f"Feature counts : {feature_counts}")
                    #     feature_values = [(feat, count, selection_id, dataset_id) for feat, count in feature_counts.items()]
                    #     print(f"Feature counts : {feature_values}")
                    #     execute_values(cursor, feature_counts_query, feature_values)

                    # Insert performance metrics
                    performance_metrics_query = """
                        INSERT INTO Performance_Metrics (classifier_id, dataset_id, matthews_corrcoef, roc_auc, accuracy, 
                            balanced_accuracy, recall, precision, f1_score, specificity)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    # Convert each metric to a list if it's an ndarray, then JSON serialize it
                    metrics = [
                        json.dumps(metric.tolist() if isinstance(metric, np.ndarray) else metric)
                        for metric in [row.get(metric) for metric in extra_metrics]
                    ]
                    print(f"Metrics : {metrics}")
                    cursor.execute(
                        performance_metrics_query,
                        [classifier_id, dataset_id] + metrics
                    )

                    # Insert samples classification rates
                    samples_classification_query = """
                        INSERT INTO Samples_Classification_Rates (classifier_id, dataset_id, samples_classification_rates)
                        VALUES (%s, %s, %s)
                    """
                    # Ensure samples classification rates are JSON serializable
                    samples_classification_rates = row["Samples_classification_rates"]
                    print(f"Samples classification rates : {samples_classification_rates}")
                    cursor.execute(
                        samples_classification_query,
                        (classifier_id, dataset_id, json.dumps(samples_classification_rates))
                    )

                # Commit the transaction
                connection.commit()
                print("Data inserted into the database successfully.")
            except Exception as e:
                connection.rollback()
                print(f"An error occurred while inserting data into the database: {e}")
            finally:
                cursor.close()
                connection.close()
                
        return statistics_dataframe