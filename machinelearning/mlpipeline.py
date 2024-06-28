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
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

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


def scoring_check(scoring: str) -> None:
    if scoring not in sklearn.metrics.get_scorer_names():
        raise ValueError(
            f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
        )


class MLPipelines(MachineLearningEstimator):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        super().__init__(label, csv_dir, estimator, param_grid)
        self.config_rncv = {}

    def _set_result_csv_name(self, csv_dir):
        """This function is used to set the name of the result nested cv file with respect to the dataset name"""
        data_name = os.path.basename(csv_dir).split(".")[0]
        return data_name

    def _create_model_instance(self, model_name, params):
        """This function creates a model instance with the given parameters
        It is used in order to prevent fittinf of an already fitted model from previous runs"""

        if model_name == "RandomForestClassifier":
            return RandomForestClassifier(**params)
        elif model_name == "LogisticRegression":
            return LogisticRegression(**params)
        elif model_name == "XGBClassifier":
            return XGBClassifier(**params)
        elif model_name == "LGBMClassifier":
            return LGBMClassifier(**params)
        elif model_name == "CatBoostClassifier":
            return CatBoostClassifier(**params)
        elif model_name == "SVC":
            return SVC(**params)
        elif model_name == "KNeighborsClassifier":
            return KNeighborsClassifier(**params)
        elif model_name == "LinearDiscriminantAnalysis":
            return LinearDiscriminantAnalysis(**params)
        elif model_name == "GaussianNB":
            return GaussianNB(**params)
        elif model_name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(**params)
        elif model_name == "GaussianProcessClassifier":
            return GaussianProcessClassifier(**params)
        elif model_name == "ElasticNet":
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    # TODO: Complete _one_sem_model function
    def _one_sem_model(self, trials, model_name):
        """This function selects the 'simplest' hyperparameters for the given model."""

        constraints = hyper_compl[model_name]

        # Find the attributes of the trials that are related to the constraints
        inner_cv_splits = self.config_rncv["inner_splits"]
        trials_data = [
            {
                "params": t.params,
                "value": t.values[0],
                "sem": t.user_attrs["std_test_score"] / (inner_cv_splits**0.5),
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

        def __model_complexity(params):
            """
            Model complexity takes into account the complex parameters of each estimator.
            Normalizes them and finds the smallest complexity level of the filtered trials.
            This way, the parameters have equal weights and we seak for a generally simpler model and not a model with the smallest (important) parameter.
            """
            complexity = 0
            for p in constraints:
                if p in params and p in optuna_grid["param_ranges"]:
                    min_val, max_val = optuna_grid["param_ranges"][p]
                    # normalize the parameter value in order to equally consider each one of them
                    normalized_value = (params[p] - min_val) / (max_val - min_val)
                    if constraints[p]:
                        # increasing with ascending complexity parameters
                        complexity += normalized_value
                    else:
                        # decreasing with ascending complexity parameters needs mirrored normalization.
                        # smallest value -> highest complexity level.
                        mirrored_value = 1 - params[p]
                        complexity += mirrored_value
            return complexity

        # Find the parameters that has the minimum complexity score in the filtered trials
        simplest_model = min(
            filtered_trials, key=lambda x: __model_complexity(x["params"])
        )
        # Return the parameters of the simplest model
        return simplest_model["params"]

    def _filter_features(self, train_index, test_index, X, y, num_feature2_use):
        """
        This function filters the features using the selected model.
        """

        # Find the train and test sets
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y[train_index], y[test_index]

        X_train, X_test = self.normalize(
            X=X_tr,
            train_test_set=True,
            X_test=X_te,
            method=self.config_rncv["normalization"],
        )

        # Manipulate the missing values for both train and test sets
        X_train = self.missing_values(
            data=X_train, method=self.config_rncv["missing_values_method"]
        )
        X_test = self.missing_values(
            data=X_test, method=self.config_rncv["missing_values_method"]
        )

        # Find the feature selection type and apply it to the train and test sets
        if self.config_rncv["feature_selection_type"] != "percentile":
            if isinstance(num_feature2_use, int):
                if num_feature2_use < X_train.shape[1]:
                    self.selected_features = self.feature_selection(
                        X=X_train,
                        y=y_train,
                        method=self.config_rncv["feature_selection_type"],
                        num_features=num_feature2_use,
                        inner_method=self.config_rncv["feature_selection_method"],
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
        elif self.config_rncv["feature_selection_type"] == "percentile":
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
                        method=self.config_rncv["feature_selection_type"],
                        inner_method=self.config_rncv["feature_selection_method"],
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

    def inner_loop(self, train_index, test_index, X, y, avail_thr):
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
            "Scores": [],
            "Classifiers": [],
            "Selected_Features": [],
            "Number_of_Features": [],
            "Hyperparameters": [],
            "Way_of_Selection": [],
            "Estimator": [],
        }

        # Loop over the number of features
        for num_feature2_use in feature_loop:
            X_train_selected, X_test_selected, num_feature = self._filter_features(
                train_index, test_index, X, y, num_feature2_use
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

                    self._set_optuna_verbosity(logging.ERROR)
                    clf = optuna.integration.OptunaSearchCV(
                        estimator=self.estimator,
                        scoring=self.config_rncv["inner_scoring"],
                        param_distributions=optuna_grid[opt_grid][self.name],
                        cv=self.config_rncv["inner_cv"],
                        n_jobs=n_jobs,
                        verbose=0,
                        n_trials=self.config_rncv["n_trials_ncv"],
                    )
                    clf.fit(X_train_selected, y_train)

                    # Store the results and apply one_sem method if its selected
                    results["Estimator"].append(self.name)
                    if self.config_rncv["inner_selection"] == "validation_score":
                        results["Scores"].append(
                            get_scorer(self.config_rncv["outer_scoring"])(
                                clf, X_test_selected, y_test
                            )
                        )
                        results["Hyperparameters"].append(clf.best_params_)
                    else:
                        trials = clf.trials_
                        # Find simpler parameters with the one_sem method if there are any
                        simple_model_params = self._one_sem_model(trials, self.name)
                        results["Hyperparameters"].append(simple_model_params)
                        # Fit the new model
                        new_params_clf = self._create_model_instance(
                            self.name, simple_model_params
                        )
                        new_params_clf.fit(X_train_selected, y_train)
                        results["Scores"].append(
                            new_params_clf.score(X_test_selected, y_test)
                        )
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

        time.sleep(1)

        return [results]

    def outer_loop(self, i, avail_thr):
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
                    results = self.inner_loop(
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
                    results = self.inner_loop(
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
        n_trials_ncv=25,
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
        inner_selection="validation_score",
        outer_scoring="matthews_corrcoef",
        inner_splits=5,
        outer_splits=5,
        normalization="minmax",
        parallel="thread_per_round",
        missing_values_method="median",
        frfs=None,
        name_add=None,
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
        if inner_selection not in ["validation_score", "one_sem"]:
            raise ValueError(
                f'Invalid inner method: {inner_selection}. Select one of the following: ["validation_score", "one_sem"]'
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
                    delayed(self.outer_loop)(i, avail_thr) for i in trial_indices
                )
        elif parallel == "freely_parallel":
            with threadpool_limits():
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(self.outer_loop)(i, avail_thr) for i in trial_indices
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

        for classif in np.unique(df["Classifiers"]):
            indices = df[df["Classifiers"] == classif]
            filtered_scores = indices["Scores"].values
            if num_features is not None:
                filtered_features = indices["Selected_Features"].values
            mean_score = np.mean(filtered_scores)
            max_score = np.max(filtered_scores)
            std_score = np.std(filtered_scores)
            sem_score = sem(filtered_scores)
            median_score = np.median(filtered_scores)
            Numbers_of_Features = indices["Number_of_Features"].unique()[0]
            Way_of_Selection = indices["Way_of_Selection"].unique()[0]
            results.append(
                {
                    "Estimator": df[df["Classifiers"] == classif]["Estimator"].unique()[
                        0
                    ],
                    "Classifier": classif,
                    "Scores": filtered_scores.tolist(),
                    "Max": max_score,
                    "Std": std_score,
                    "SEM": sem_score,
                    "Median": median_score,
                    "Hyperparameters": df[df["Classifiers"] == classif][
                        "Hyperparameters"
                    ].values,
                    "Selected_Features": filtered_features
                    if num_features is not None
                    else None,
                    "Numbers_of_Features": Numbers_of_Features,
                    "Way_of_Selection": Way_of_Selection,
                }
            )

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
            dataset_name = self.set_result_csv_name(self.csv_dir)
            if name_add is None:
                results_name = f"{dataset_name}_{inner_selection}_{features}"
            else:
                results_name = f"{dataset_name}_{inner_selection}_{features}_{name_add}"
            final_dataset_name = os.path.join(results_dir, results_name)
        except Exception as e:
            dataset_name = "dataset"
            if name_add is None:
                results_name = f"{dataset_name}_{inner_selection}_{features}"
            else:
                results_name = f"{dataset_name}_{inner_selection}_{features}_{name_add}"
            final_dataset_name = os.path.join(
                results_dir, f"{dataset_name}_{inner_selection}_{features}"
            )

        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:
            if freq_feat == None:
                freq_feat = self.X.shape[1]
            elif freq_feat > self.X.shape[1]:
                freq_feat = self.X.shape[1]

            # Plot histogram of features
            feature_counts = Counter()
            for idx, row in scores_dataframe.iterrows():
                if (
                    row["Way_of_Selection"] != "full"
                ):  # If no features were selected skip
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
                counts = [x / len(clfs) for x in counts]
                plt.figure(figsize=(max(10, freq_feat // 2), 10))
                bars = plt.bar(features, counts, color="skyblue", tick_label=features)
                plt.xlabel("Features")
                plt.ylabel("Counts")
                plt.title("Histogram of Selected Features")
                plt.xticks(rotation=90)

                plt.gca().margins(x=0.05)
                plt.gcf().canvas.draw()
                tl = plt.gca().get_xticklabels()
                maxsize = max([t.get_window_extent().width for t in tl])
                m = 0.5
                s = maxsize / plt.gcf().dpi * freq_feat + 2 * m
                margin = m / plt.gcf().get_size_inches()[0]

                plt.gcf().subplots_adjust(left=margin, right=1.0 - margin)
                plt.gca().set_xticks(plt.gca().get_xticks()[::1])
                plt.gca().set_xlim([-1, freq_feat])

                plt.tight_layout()
                plt.show()

                # Save the plot to 'Results/histogram.png'
                save_path = f"{final_dataset_name}_histogram.png"
                plt.savefig(save_path, bbox_inches="tight")

                # Show the plot
                plt.show()

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

        # Plot box or violin plots of the outer cross-validation scores
        if plot is not None:
            scores_long = scores_dataframe.explode("Scores")
            scores_long["Scores"] = scores_long["Scores"].astype(float)

            fig = go.Figure()
            if plot == "box":
                # Add box plots for each classifier
                for classifier in scores_dataframe["Classifier"]:
                    data = scores_long[scores_long["Classifier"] == classifier][
                        "Scores"
                    ]
                    fig.add_trace(
                        go.Box(
                            y=data,
                            name=classifier,
                            boxpoints="all",
                            jitter=0.3,
                            pointpos=-1.8,
                        )
                    )
                    # Calculate and add 95% CI for the median
                    lower, upper = bootstrap_median_ci(data)
                    fig.add_trace(
                        go.Scatter(
                            x=[classifier, classifier],
                            y=[lower, upper],
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                            showlegend=False,
                        )
                    )

            elif plot == "violin":
                for classifier in scores_dataframe["Classifier"]:
                    data = scores_long[scores_long["Classifier"] == classifier][
                        "Scores"
                    ]
                    fig.add_trace(
                        go.Violin(
                            y=data,
                            name=classifier,
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
                yaxis_title="Score",
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

        # Save the results to a CSV file of the outer scores for each classifier
        if return_csv:
            results_path = f"{final_dataset_name}_outerloops_results.csv"
            scores_dataframe.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")

        # Return the dataframe and the list of features if feature selection is applied
        if num_features is not None:
            return scores_dataframe, features_list
        else:
            return scores_dataframe