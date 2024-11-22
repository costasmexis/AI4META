# Standard library imports
import os
import time
import json
import logging
import multiprocessing
from typing import Union
import copy
from collections import Counter
from itertools import chain

# Numerical computing
import numpy as np
import pandas as pd
from scipy.stats import sem

# Machine learning and data processing
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer, confusion_matrix, make_scorer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

# Optimization and multiprocessing
import optuna
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

# Custom modules
from .mlestimator import MachineLearningEstimator
from .optuna_grid import optuna_grid, hyper_compl
from .utils import pipelines_utils

# Progress tracking
import progressbar

class MLPipelines(MachineLearningEstimator):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        super().__init__(label, csv_dir, estimator, param_grid)
        self.config_rncv = {}


    def _gso_model(self, trials, model_name, splits, method):
        """
        This function selects the 'balanced trainned' hyperparameters for the given model.
        
        The 'balanced trainned' hyperparameters are defined as the hyperparameters that have
        the best train or validation score and a score threshold that is 85% of the
        best score. The hyperparameters are selected by sorting the trials by score
        and then filtering the trials by those that are above the score threshold.
        The trial with the smallest average gap score is then selected.
        
        Parameters
        ----------
        trials : list of optuna.trial.Trial
            List of trials to select the best hyperparameters from
        model_name : str
            Name of the model to select the hyperparameters for
        splits : int
            Number of splits to calculate the gap score
        method : str
            Method to select the hyperparameters. Can be 'gso_1' or 'gso_2'
        
        Returns
        -------
        dict
            The selected hyperparameters as a dictionary
        """
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
        """
        This function selects the 'simplest' hyperparameters for the given model.
        It does this by selecting the hyperparameters that resulted validation score higher than the SEM threshold and have the smallest complexity score which is calculated differently in each estimator.
        
        Parameters
        ----------
        trials : list of optuna.trial.Trial
            List of trials to select from.
        model_name : str
            Name of the model to select hyperparameters for.
        samples : int
            Number of samples in the dataset.
        splits : int
            Number of splits in the cross-validation.
        method : str
            Method to use for selecting the simplest hyperparameters. Can be 'one_sem' or 'one_sem_grd'.
        
        Returns
        -------
        dict
            Selected hyperparameters.
        """
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
            # Calculate complexity for each filtered trial
            for trial in filtered_trials:
                trial["complexity"] = pipelines_utils._calculate_complexity(trial, model_name, samples)

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

    def _cv_loop(self, i, avail_thr):
        """
        This function is used to perform the rounds loop of the cross-validation using only the default parameters. 
        It is useful for first insights about the estimators performance and is fast for big datasets.
        It is used only in rcv_accel function.

        Parameters
        ----------
        i : int
            The current round of the cross-validation.

        avail_thr : int
            The number of available threads for parallelization.

        Returns
        -------
        list_dfs : list of pandas DataFrames
            A list of dataframes containing the results of the cross-validation.
        """
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
                                X_train_selected, X_test_selected, num_feature = pipelines_utils._sfm(self.estimator, X_train_selected, X_test_selected, y_train, num_feature2_use)

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
                            clf = pipelines_utils._create_model_instance(
                                    self.name, params=None
                                )
                            clf.fit(X_train_selected, y_train)
                            
                            # Store the results and apply one_sem method if its selected
                            results["Estimator"].append(self.name)
                            for metric in self.config_rcv["extra_metrics"]:
                                if metric == 'specificity':
                                    results[f"{metric}"].append(
                                        pipelines_utils._specificity_scorer(clf, X_test_selected, y_test)
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
        class_balance = 'auto',
        sfm=False,
        extra_metrics=['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','average_precision','specificity'],
        info_to_db=False,
        filter_csv=None
    ):
        """
        Accelerated version of the nested cross-validation process.

        This function performs nested cross-validation faster by using parallel
        processing. It is the same as the nested cross-validation process, but
        it uses the joblib library to use multiple cores of the computer.

        Parameters
        ----------
        rounds : int, optional
            Number of rounds to perform the nested cross-validation. The
            default is 10.
        exclude : list, optional
            List of estimators to exclude from the nested cross-validation.
            The default is None.
        search_on : str, optional
            Estimator to search for hyperparameters. The default is None.
        num_features : int, optional
            Number of features to select. The default is None.
        feature_selection_type : str, optional
            Method to select features. The default is 'mrmr'.
        return_csv : bool, optional
            Whether to return the results as a CSV file. The default is True.
        feature_selection_method : str, optional
            Method to select features. The default is 'chi2'.
        plot : str, optional
            Whether to plot the results. The default is 'box'.
        scoring : str, optional
            Scoring method to use. The default is 'matthews_corrcoef'.
        splits : int, optional
            Number of splits to perform in the nested cross-validation. The
            default is 5.
        normalization : str, optional
            Method of normalization, either 'minmax' or 'std' or None, by default 'minmax'
        missing_values_method : str, optional
            Method of handling missing values, either 'median' or 'mean' or '0' or 'drop' or None, by default 'median'
        num_features : int|list, optional
            Number of features to select, either a single number or a list of numbers, by default None (all features)
        feature_selection_type : str, optional
            Type of feature selection, either 'mrmr' or 'kbest' or 'percentile', by default 'mrmr' (if no number of features is provided its not in use)
        feature_selection_method : str, optional
            Method of feature selection, either 'chi2', 'mutual_info_classif', 'f_classif', by default 'chi2' (if no number of features is provided its not in use)
        sfm : bool, optional
            If True, the feature selection is done using the SelectFromModel class only for the estimators that support it. The rest of the estimators are using the feature_selection_type. By default False
        freq_feat : int, optional
            A histogram of the frequency of the (freaq_feat or all if None) features will be plotted, by default None
        class_balance : str, optional
            If 'auto', class balancing will be applied using 'smote', 'smote_enn', 'adasyn', 'borderline_smote', or 'tomek'. By default 'auto'
        extra_metrics : list, optional
            List of extra metrics to calculate. The default is
            ['roc_auc','accuracy','balanced_accuracy','recall','precision',
            'f1','average_precision','specificity'].
        info_to_db : bool, optional
            Whether to add the results to a database. The default is False.
        filter_csv : list, optional
            List of columns to filter from the CSV file. The default is None.

        Returns
        -------
        results : pandas dataframe
            Dataframe with the results of the nested cross-validation.
        """
        self.config_rcv = locals()
        self.config_rcv.pop("self", None)
        self.config_rcv = self._parameters_check(self.config_rcv,'rcv')
        
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
                    "Class_bal": self.config_rcv['class_balance'],
                    "Scoring": scoring,
                    "In_sel": 'validation_score',
                    "Classif_rates": samples_classification_rates.tolist(),
                }
            )
            
            results = pipelines_utils._input_renamed_metrics(
                self.config_rcv['extra_metrics'], results, indices
            )
                                            
        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)

        # Create a 'Results' directory
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Initiate name
        final_dataset_name = pipelines_utils._name_outputs(self.config_rcv, results_dir, self.csv_dir)  
          
        # Save the results to a CSV file of the outer scores for each classifier
        if return_csv:
            statistics_dataframe = pipelines_utils._return_csv(final_dataset_name, scores_dataframe, self.config_rcv['extra_metrics'], filter_csv)

        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:
            # Plot histogram of features
            pipelines_utils._histogram(scores_dataframe, final_dataset_name, freq_feat, self.config_rcv['clfs'], self.X.shape[1])
        
        # Plot box or violin plots of the outer cross-validation scores 
        if plot is not None:
            pipelines_utils._plot(scores_dataframe, plot, self.config_rcv['scoring'], final_dataset_name)
            
        if info_to_db:
            # Add to database
            pipelines_utils._insert_data_into_db(scores_dataframe, self.config_rcv)

        return statistics_dataframe
            
    def _filter_features(self, train_index, test_index, X, y, num_feature2_use, cvncvsel):
        """
        This function filters the features using the selected model.
        Returns the filtered train and test sets indexes - including the normalized data and the missing values - and the selected features.
        
        Parameters
        ----------
        train_index : array-like
            The indices of the samples in the training set.
        test_index : array-like
            The indices of the samples in the test set.
        X : pandas DataFrame
            The features of the dataset.
        y : pandas Series
            The target variable of the dataset.
        num_feature2_use : int
            The number of features to select.
        cvncvsel : str
            Whether to use the config for nested cv or recursive nested cv.
        
        Returns
        -------
        X_train_selected : pandas DataFrame
            The filtered training set.
        X_test_selected : pandas DataFrame
            The filtered test set.
        num_feature : int or str
            The number of features selected or "full" if all features were selected.
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
        """
        This function is used to perform the inner loop of the nested cross-validation for the selection of the best hyperparameters.
        Note: Return a list because this is the desired output for the parallel loop
        Runs only on the nested cross-validation function.
        Supports several inner selection methods.
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
                        # Check of the classifiers given list
                        # Perform feature selection using Select From Model (sfm)
                        X_train_selected, X_test_selected, num_feature = pipelines_utils._sfm(self.estimator, X_train_selected, X_test_selected, y_train, num_feature2_use)

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

                            trials = clf.trials_
                        else:
                            if (inner_selection == "one_sem") or (inner_selection == "one_sem_grd"):
                                samples = X_train_selected.shape[0]
                                # Find simpler parameters with the one_sem method if there are any
                                simple_model_params = self._one_sem_model(trials, self.name, samples, self.config_rncv['inner_splits'],inner_selection)
                            elif (inner_selection == "gso_1") or (inner_selection == "gso_2"):
                                # Find parameters with the smaller gap score with gso_1 method if there are any
                                simple_model_params = self._gso_model(trials, self.name, self.config_rncv['inner_splits'],inner_selection)

                            params = simple_model_params

                            # Fit the new model
                            new_params_clf = pipelines_utils._create_model_instance(
                                self.name, simple_model_params
                            )
                            new_params_clf.fit(X_train_selected, y_train)

                            res_model = copy.deepcopy(new_params_clf)
                            
                        results["Hyperparameters"].append(params)

                        # print('Inner selection finished')
                        
                        for metric in self.config_rncv["extra_metrics"]:
                            if metric == 'specificity':
                                results[f"{metric}"].append(
                                    pipelines_utils._specificity_scorer(res_model, X_test_selected, y_test)
                                )
                            else:
                                print(res_model)
                                try:                                 
                                    results[f"{metric}"].append(
                                        get_scorer(metric)(res_model, X_test_selected, y_test)
                                    )
                                except AttributeError:
                                    # Handle metrics like roc_auc and average_precision explicitly
                                    if metric in ['roc_auc', 'average_precision']:
                                        if hasattr(res_model, 'predict_proba'):
                                            # Use decision_function if available
                                            y_pred = res_model.predict_proba(X_test_selected)[:, 1]
                                        elif hasattr(res_model, 'decision_function'):
                                            # Use predict_proba for metrics requiring probabilities
                                            y_pred = res_model.decision_function(X_test_selected)
                                        else:
                                            raise AttributeError(
                                                f"Model {type(res_model).__name__} does not support `predict_proba` or `decision_function`, "
                                                f"which are required for {metric}."
                                            )

                                        # Compute the score using the selected y_pred
                                        if metric == 'roc_auc':
                                            score = roc_auc_score(y_test, y_pred)
                                        elif metric == 'average_precision':
                                            score = average_precision_score(y_test, y_pred)

                                    results[f"{metric}"].append(score)
                        
                        # print('Before prediction')
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
        """
        This function is used to make the separations of train and test data of the outer cross validation and initiates the parallilization
        with respect to the parallelization method.
        Uses different random seed for each round of the outer cross validation
        """
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

    def _parameters_check(self, config, main_type):
        """
        This function checks the parameters of the pipeline and returns the final parameters config for the class pipeline.
        """
        # Missing values manipulation
        if config['missing_values_method'] == "drop":
            print(
                "Values cannot be dropped at ncv because of inconsistent shapes. \nThe missing values with automaticly replaced by the median of each feature."
            )
            config['missing_values_method'] = "median"
        elif (config['missing_values_method'] != "mean") and (config['missing_values_method'] != "median"):
            raise ValueError(
                "The missing values method should be 'mean' or 'median'."
            )
        if self.X.isnull().values.any():
            print(
                f"Your Dataset contains NaN values. Some estimators does not work with NaN values.\nThe {config['missing_values_method']} method will be used for the missing values manipulation.\n"
            )
        
        if config['extra_metrics'] is not None:
            if type(config['extra_metrics']) is not list:
                config['extra_metrics'] = [config['extra_metrics']]
            for metric in config['extra_metrics']:
                pipelines_utils._scoring_check(metric)
            
        if main_type == 'rncv':    
            if config['outer_scoring'] not in config['extra_metrics']:
                config['extra_metrics'].insert(0, config['outer_scoring'])
            elif config['outer_scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['outer_scoring']) != 0:
                # Remove it from its current position
                config['extra_metrics'].remove(config['outer_scoring'])
                # Insert it at the first index
                config['extra_metrics'].insert(0, config['outer_scoring'])
            elif config['outer_scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['outer_scoring']) == 0:
                pass
            else:
                config['extra_metrics'] = [config['outer_scoring']]
                
            config['model_selection_type'] = 'rncv'
            
            # Checks for reliability of parameters
            if (config['inner_scoring'] not in sklearn.metrics.get_scorer_names()) and (config['inner_scoring'] != "specificity"):
                raise ValueError(
                    f"Invalid inner scoring metric: {config['inner_scoring']}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
                )
            if (config['outer_scoring'] not in sklearn.metrics.get_scorer_names()) and (config['outer_scoring'] != "specificity"):
                raise ValueError(
                    f"Invalid outer scoring metric: {config['outer_scoring']}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
                )
            for inner_selection in config['inner_selection_lst']:
                if inner_selection not in ["validation_score", "one_sem", "gso_1", "gso_2","one_sem_grd"]:
                    raise ValueError(
                        f'Invalid inner method: {inner_selection}. Select one of the following: ["validation_score", "one_sem", "one_sem_grd", "gso_1", "gso_2"]'
                    )
                    
            if (config['parallel'] not in ['thread_per_round', 'freely_parallel']) and (config['parallel'] is not None):
                raise ValueError(
                    f'Invalid parallel method: {config["parallel"]}. Select one of the following: ["thread_per_round", "freely_parallel"]'
            )
            elif (config['parallel'] == None):
                config['parallel'] = 'thread_per_round'
                print('Parallel method is set to "thread_per_round"')
            
        else:
            if config['scoring'] not in config['extra_metrics']:
                config['extra_metrics'].insert(0, config['scoring'])
            elif config['scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['scoring']) != 0:
                # Remove it from its current position
                config['extra_metrics'].remove(config['scoring'])
                # Insert it at the first index
                config['extra_metrics'].insert(0, config['scoring'])
            elif config['scoring'] in config['extra_metrics'] and config['extra_metrics'].index(config['scoring']) == 0:
                pass
            else:
                config['extra_metrics'] = [config['scoring']]
            
            config['model_selection_type'] = 'rcv'
            
            if (config['scoring'] not in sklearn.metrics.get_scorer_names()) and (config['scoring'] != "specificity"):
                raise ValueError(
                    f"Invalid scoring metric: {config['scoring']}. Select one of the following: {list(sklearn.metrics.get_scorer_names())} and specificity"
                )
                
        if config['class_balance'] not in ['auto','smote','smote_enn','adasyn','borderline_smote','tomek', None]:
            raise ValueError("class_balance must be one of the following: 'auto','smote','smotenn','adasyn','borderline_smote','tomek', or None")
        elif config['class_balance'] == None:
            config['class_balance'] = 'auto'
            print('Class balance is set to "auto"')
            
        config['dataset_name'] = self.csv_dir
                
        # Set available classifiers
        if config['search_on'] is not None:
            classes = config['search_on']  # 'search_on' is a list of classifier names as strings
            exclude_classes = [
                clf for clf in self.available_clfs.keys() if clf not in classes
            ]
        elif config['exclude'] is not None:
             exclude_classes = (
                config['exclude']  # 'exclude' is a list of classifier names as strings
            )
        else:
            exclude_classes = []

        # Filter classifiers based on the exclude_classes list
        clfs = [clf for clf in self.available_clfs.keys() if clf not in exclude_classes]
        config["clfs"] = clfs

        return config
    
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
        Perform nested cross-validation with optional feature selection and hyperparameter optimization.

        Parameters
        ----------
        n_trials : int, optional
            Number of optuna trials for hyperparameter optimization, by default 100
        rounds : int, optional
            Number of outer cross-validation rounds, by default 10
        exclude : str|list, optional
            List of classifiers (or string with one classifier) to exclude from the search, by default None
        search_on : str|list, optional
            List of classifiers (or string with one classifier) to search for hyperparameters on, by default None
        info_to_db : bool, optional
            If True, the results will be added to the database, by default False
        num_features : int|list, optional
            Number of features to select, either a single number or a list of numbers, by default None (all features)
        feature_selection_type : str, optional
            Type of feature selection, either 'mrmr' or 'kbest' or 'percentile', by default 'mrmr' (if no number of features is provided its not in use)
        feature_selection_method : str, optional
            Method of feature selection, either 'chi2', 'mutual_info_classif', 'f_classif', by default 'chi2' (if no number of features is provided its not in use)
        sfm : bool, optional
            If True, the feature selection is done using the SelectFromModel class only for the estimators that support it. The rest of the estimators are using the feature_selection_type. By default False
        freq_feat : int, optional
            A histogram of the frequency of the (freaq_feat or all if None) features will be plotted, by default None
        class_balance : str, optional
            If 'auto', class balancing will be applied using 'smote', 'smote_enn', 'adasyn', 'borderline_smote', or 'tomek'. By default 'auto'
        inner_scoring : str, optional
            Scoring metric used in the inner cross-validation loop, by default 'matthews_corrcoef'
        outer_scoring : str, optional
            Scoring metric used in the outer cross-validation loop, by default 'matthews_corrcoef'
        inner_selection_lst : list, optional
            List of methods for selecting the hyperparameters, by default ['validation_score', 'one_sem', 'gso_1', 'gso_2', 'one_sem_grd']
            Should be a subset of ['validation_score', 'one_sem', 'gso_1', 'gso_2', 'one_sem_grd'] to be valid.
        extra_metrics : str|list, optional
            List of extra metrics (or string with one metric) to calculate, by default ['recall', 'specificity', 'accuracy', 'balanced_accuracy', 
            'precision', 'f1', 'roc_auc', 'average_precision', 'matthews_corrcoef']
        plot : str, optional
            Type of plot to create, either 'box' or 'violin', by default 'box'
        inner_splits : int, optional
            Number of splits in the inner cross-validation loop, by default 5
        outer_splits : int, optional
            Number of splits in the outer cross-validation loop, by default 5
        parallel : str, optional
            Method of parallelization, either 'thread_per_round' or 'freely_parallel', by default 'thread_per_round'
            If the user needs resources for other procedures, it is recommended to use 'thread_per_round'
        normalization : str, optional
            Method of normalization, either 'minmax' or 'std' or None, by default 'minmax'
        missing_values_method : str, optional
            Method of handling missing values, either 'median' or 'mean' or '0' or 'drop' or None, by default 'median'
        return_csv : bool, optional
            If True, the results will be returned as a CSV file with the statistics of the results, by default True
        filter_csv : dict, optional
            Dictionary of filters to apply to the trials of the results dataframe and csv, by default None

        Returns
        -------
        pandas.DataFrame
            A dataframe with the results of the nested cross-validation.
        
            An updated database if info_to_db is True
            Plots if plot is not None
            Histogram of the frequency of the features if freq_feat is not None
        """
        self.config_rncv = locals()
        self.config_rncv.pop("self", None)
        self.config_rncv = self._parameters_check(self.config_rncv,'rncv')

        # Parallelization
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        if num_cores < rounds:
            use_cores = num_cores
        else:
            use_cores = rounds
        avail_thr = max(1, num_cores // rounds)

        if self.config_rncv['parallel'] == "thread_per_round":
            avail_thr = 1
            with threadpool_limits(limits=avail_thr):
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(self._outer_loop)(i, avail_thr) for i in trial_indices
                )
        else: 
            with threadpool_limits():
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(self._outer_loop)(i, avail_thr) for i in trial_indices
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
                        "Class_blnc": self.config_rncv['class_balance'],
                        "In_scor": inner_scoring,
                        "Out_scor": outer_scoring,
                        "In_sel":inner_selection,
                        "Classif_rates": samples_classification_rates.tolist(),
                    }
                )

                results = pipelines_utils._input_renamed_metrics(
                    self.config_rncv['extra_metrics'], results, indices
                )
                                            
        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)
        
        # Create a 'Results' directory
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Name
        final_dataset_name = pipelines_utils._name_outputs(self.config_rncv, results_dir, self.csv_dir)  
            
        # Save the results to a CSV file of the outer scores for each classifier
        if return_csv:
            statistics_dataframe = pipelines_utils._return_csv(final_dataset_name, scores_dataframe, self.config_rncv['extra_metrics'], filter_csv)
            
        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:    
            # Plot histogram of features
            pipelines_utils._histogram(scores_dataframe, final_dataset_name, freq_feat, self.config_rncv['clfs'], self.X.shape[1])

        
        # Plot box or violin plots of the outer cross-validation scores for all Inner_Selection methods
        if plot is not None:
            pipelines_utils._plot(scores_dataframe, plot, self.config_rncv['outer_scoring'], final_dataset_name)
            
        if info_to_db:
            # Add to database
            pipelines_utils._insert_data_into_db(scores_dataframe, self.config_rncv)
            
        return statistics_dataframe