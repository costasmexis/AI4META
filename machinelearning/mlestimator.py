import copy
import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import sklearn
import sklearn.metrics as metrics
from catboost import CatBoostClassifier
from dataloader import DataLoader
from joblib import Parallel, delayed, parallel_backend
from lightgbm import LGBMClassifier
from optuna.samplers import RandomSampler, TPESampler
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, matthews_corrcoef
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
from sklearn.utils import resample
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from xgboost import XGBClassifier

from .optuna_grid import optuna_grid


def scoring_check(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError:
            if kwargs["scoring"] not in sklearn.metrics.get_scorer_names():
                raise ValueError(
                    f"Invalid scoring metric: {kwargs['scoring']}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
                )

    return wrapper


class MachineLearningEstimator(DataLoader):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        """
        Initialize the MLEstimator object.

        :param label: The label for the estimator.
        :type label: str
        :param csv_dir: The directory containing the CSV files.
        :type csv_dir: str
        :param estimator: The estimator object to use, defaults to None.
        :type estimator: object, optional
        :param param_grid: The parameter grid for hyperparameter tuning, defaults to None.
        :type param_grid: dict, optional
        """

        super().__init__(label, csv_dir)
        self.estimator = estimator
        self.name = estimator.__class__.__name__
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None

        self.available_clfs = {
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
            "LogisticRegression": LogisticRegression(),
            "XGBClassifier": XGBClassifier(),
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "SVC": SVC(),
            "LGBMClassifier": LGBMClassifier(),
            "GaussianProcessClassifier": GaussianProcessClassifier(),
            "CatBoostClassifier": CatBoostClassifier(),
        }

        # Checking if estimator is valid
        if self.estimator is not None:
            if self.name not in self.available_clfs.keys():
                raise ValueError(
                    f"Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}"
                )
        elif self.estimator is None:
            print("There is no selected classifier.")

    @staticmethod
    def set_optuna_verbosity(level):
        """
        Set the verbosity level for Optuna.

        :param level: The verbosity level.
        :type level: int
        """
        optuna.logging.set_verbosity(level)
        logging.getLogger("optuna").setLevel(level)

    @scoring_check
    def grid_search(
        self,
        X=None,
        y=None,
        estimator=None,
        parameter_grid=None,
        scoring="matthews_corrcoef",
        features_list=None,
        feat_num=None,
        feat_way="mrmr",
        cv=5,
        verbose=True,
        return_model=False,
    ):
        """
        Perform grid search to find the best hyperparameters for the estimator.

        :param X: The input features, defaults to None
        :type X: array-like, shape (n_samples, n_features), optional
        :param y: The target values, defaults to None
        :type y: array-like, shape (n_samples,), optional
        :param estimator: The estimator object. If None self.estimator is selected, defaults to None.
        :type estimator: object, optional
        :param parameter_grid: The parameter grid for hyperparameter tuning, defaults to None
        :type parameter_grid: dict, optional
        :param scoring: The scoring metric to optimize, defaults to 'matthews_corrcoef'
        :type scoring: str, optional
        :param features_list: List of feature names to use for grid search, defaults to None
        :type features_list: list, optional
        :param feat_num: Number of features to select using feature selection method, defaults to None
        :type feat_num: int, optional
        :param feat_way: The feature selection method to use, defaults to 'mrmr'
        :type feat_way: str, optional
        :param cv: The number of cross-validation folds, defaults to 5
        :type cv: int, optional
        :param verbose: Whether to print the best parameters and score, defaults to True
        :type verbose: bool, optional
        :param return_model: Whether to return the best estimator, defaults to False
        :type return_model: bool, optional
        :return: The best estimator if `return_model` is True, otherwise None
        :rtype: estimator object or None
        """

        X = X or self.X
        y = y or self.y

        if len(X) != len(y):
            raise ValueError("The length of X and y must be equal.")

        if features_list is not None:
            X = X[features_list]
        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        if estimator is not None:
            self.estimator = estimator
            self.name = self.estimator.__class__.__name__
            self.param_grid = parameter_grid

        grid_search = GridSearchCV(
            self.estimator, self.param_grid, scoring=scoring, cv=cv
        )
        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        self.name = self.best_estimator.__class__.__name__

        if verbose:
            print(f"Estimator: {self.name}")
            print(f"Best parameters: {self.best_params}")
            print(f"Best {scoring}: {self.best_score}")

        if return_model:
            return self.best_estimator

    @scoring_check
    def random_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        features_list=None,
        boxplot=True,
        cv=5,
        direction="maximize",
        n_iter=10,
        estimator_name=None,
        calculate_shap=False,
        feat_num=None,
        feat_way="mrmr",
        missing_values="median",
        parameter_grid=None,
    ):
        """
        Perform a random search for hyperparameter tuning.

        :param X: The input features, defaults to None
        :type X: array-like, optional
        :param y: The target variable, defaults to None
        :type y: array-like, optional
        :param scoring: The scoring metric to optimize, defaults to "matthews_corrcoef"
        :type scoring: str, optional
        :param features_list: The list of features to use, defaults to None
        :type features_list: list, optional
        :param boxplot: Whether to plot boxplots, defaults to True
        :type boxplot: bool, optional
        :param cv: The number of cross-validation folds, defaults to 5
        :type cv: int, optional
        :param direction: The direction to optimize the scoring metric, defaults to "maximize"
        :type direction: str, optional
        :param n_iter: The number of iterations for random search, defaults to 10
        :type n_iter: int, optional
        :param estimator_name: The name of the estimator, defaults to None
        :type estimator_name: str, optional
        :param calculate_shap: Whether to calculate SHAP values, defaults to False
        :type calculate_shap: bool, optional
        :param feat_num: The number of features to select, defaults to None
        :type feat_num: int, optional
        :param feat_way: The feature selection method, defaults to "mrmr"
        :type feat_way: str, optional
        :param missing_values: The method to handle missing values, defaults to "median"
        :type missing_values: str, optional
        :param parameter_grid: The parameter grid for hyperparameter search, defaults to None
        :type parameter_grid: dict, optional
        :raises ValueError: If the length of X and y are not equal
        :raises ValueError: If a parameter grid is not provided
        :return: The best estimator, the results dataframe
        :rtype: tuple
        """

        X = X or self.X
        y = y or self.y

        if len(X) != len(y):
            raise ValueError("The length of X and y must be equal.")

        if features_list is not None:
            X = X[features_list]
        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        if estimator_name is None:
            estimator_name = self.name
        else:
            estimator_name = estimator_name

        self.estimator = self.available_clfs[estimator_name]

        if parameter_grid is None:
            raise ValueError("Please provide a parameter grid.")
        else:
            self.param_grid = parameter_grid

        cv_splits = StratifiedKFold(n_splits=cv, shuffle=True)
        temp_train_test_indices = list(cv_splits.split(X, y))

        random_search = RandomizedSearchCV(
            self.estimator,
            self.param_grid,
            scoring=scoring,
            cv=temp_train_test_indices,
            n_iter=n_iter,
        )
        random_search.fit(X, y)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_
        self.name = self.best_estimator.__class__.__name__

        print(f"Estimator: {self.name}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best {scoring}: {self.best_score}")

        results_df = pd.DataFrame.from_dict(random_search.cv_results_)
        results_df = results_df.sort_values(by="rank_test_score")
        usefull_cols = [
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "params",
        ]
        for i in range(cv):
            usefull_cols.append(f"split{i}_test_score")
        data_full_outer = results_df[usefull_cols]

        if boxplot:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            scores = []
            for i in range(cv):
                scores = np.append(scores, data_full_outer[f"split{i}_test_score"])
            plt.boxplot(scores)
            plt.title("Summary Boxplot")
            plt.ylabel("Score")
            plt.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )

            plt.subplot(1, 2, 2)
            temp_df = data_full_outer[
                (data_full_outer["rank_test_score"] == 1)
                | (data_full_outer["rank_test_score"] == data_full_outer.shape[0])
            ]
            temp_df.reset_index(drop=True, inplace=True)
            for idx, row in temp_df.iterrows():
                row_scores = []
                for i in range(cv):
                    row_scores = np.append(row_scores, row[f"split{i}_test_score"])
                plt.boxplot(row_scores, positions=[idx + 1])
                if row["rank_test_score"] == 1:
                    plt.text(
                        idx + 1,
                        row.mean_test_score,
                        f"Best trial\nmean:{row.mean_test_score:.2f}",
                        ha="center",
                        va="bottom",
                        color="red",
                    )
                else:
                    plt.text(
                        idx + 1,
                        row.mean_test_score,
                        f"Worst trial\nmean:{row.mean_test_score:.2f}",
                        ha="center",
                        va="bottom",
                        color="blue",
                    )

            plt.title(f"Best and Worst Trials for {estimator_name}")
            plt.ylabel("Score")
            plt.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            plt.show()

        if calculate_shap:
            x_shap = np.zeros((X.shape[0], X.shape[1]))
            for train_index, test_index in temp_train_test_indices:
                X_train, _ = X.iloc[train_index], X.iloc[test_index]
                y_train, _ = y[train_index], y[test_index]
                best_model = self.available_clfs[estimator_name]
                best_model.set_params(**self.best_params)
                best_model.fit(X_train, y_train)
                shap_values = self.calc_shap(X_train, best_model)
                x_shap[train_index, :] = np.add(
                    x_shap[train_index, :], shap_values.values
                )
            x_shap = x_shap / (cv - 1)
            return self.best_estimator, results_df, x_shap
        else:
            return self.best_estimator, results_df

    def calc_shap(self, X, model):
        try:
            explainer = shap.Explainer(model, X)
        except TypeError as e:
            if (
                "The passed model is not callable and cannot be analyzed directly with the given masker!"
                in str(e)
            ):
                print(
                    "Switching to predict_proba due to compatibility issue with the model."
                )
                explainer = shap.Explainer(lambda x: model.predict_proba(x), X)
            else:
                raise TypeError(e)
        try:
            shap_values = explainer(X)
        except ValueError:
            num_features = X.shape[1]
            max_evals = 2 * num_features + 1
            shap_values = explainer(X, max_evals=max_evals)

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        else:
            pass
        return shap_values

    @scoring_check
    def bayesian_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        features_list=None,
        rounds=10,
        boxplot=True,
        cv=5,
        direction="maximize",
        n_trials=100,
        estimator_name=None,
        evaluation="cv_simple",
        feat_num=None,
        feat_way="mrmr",
        verbose=True,
        missing_values="median",
        calculate_shap=False,
        param_grid=None,
    ):
        self.set_optuna_verbosity(logging.ERROR)

        X = X or self.X
        y = y or self.y

        if missing_values is not None:
            X = self.missing_values(X, method=missing_values)

        if features_list is not None:
            X = X[features_list]
        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        estimator_name = estimator_name or self.name

        estimator = self.available_clfs[estimator_name]

        if param_grid is None:
            self.param_grid = optuna_grid["ManualSearch"]
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid

        if calculate_shap:
            shaps_array = np.zeros((X.shape[0], X.shape[1]))

        self.params = locals()
        self.params.pop("self", None)

        data_full_outer = pd.DataFrame()
        if calculate_shap:
            best_model, data_full_outer, shaps_array = self.c_v()
        else:
            best_model, data_full_outer = self.c_v()

        if boxplot:
            if evaluation == "bootstrap":
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        y=data_full_outer["Scores"],
                        name=estimator_name,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                        boxmean=True,
                    )
                )

                fig.update_layout(
                    title=f"Model Evaluation Results With {evaluation} Method",
                    yaxis_title="Score",
                    template="plotly_white",
                )
            elif evaluation == "cv_simple":
                fig = go.Figure()
                all_scores = []
                best_cv_scores = []
                for i in range(cv):
                    for row in range(data_full_outer.shape[0]):
                        all_scores.append(
                            data_full_outer[f"split{i}_test_score"].iloc[row]
                        )

                fig.add_trace(
                    go.Box(
                        y=all_scores,
                        name="All trials scores",
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                        boxmean=True,
                    )
                )

                temp_best_cv = data_full_outer[data_full_outer["ranked"] == 1]
                for i in range(cv):
                    best_cv_scores.append(temp_best_cv[f"split{i}_test_score"].iloc[0])

                fig.add_trace(
                    go.Box(
                        y=best_cv_scores,
                        name="Best trial Scores",
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                        boxmean=True,
                    )
                )

                fig.update_layout(
                    title=f"Model Evaluation Results With {evaluation} Method",
                    yaxis_title="Score",
                    template="plotly_white",
                )
            else:
                fig = make_subplots(
                    rows=1, cols=2, subplot_titles=("Summary Boxplots", "Rounds Scores")
                )
                all_scores = []
                best_scores_rounds = []
                best_cv = []
                for i in range(cv):
                    for row in range(data_full_outer.shape[0]):
                        all_scores.append(
                            data_full_outer[f"split{i}_test_score"].iloc[row]
                        )

                fig.add_trace(
                    go.Box(
                        y=all_scores,
                        name="All trial Scores",
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=1,
                    col=1,
                )

                for round in data_full_outer["round"].unique():
                    rounds_df = data_full_outer[data_full_outer["round"] == round]
                    rounds_df = rounds_df.sort_values(
                        by="mean_test_score", ascending=False
                    )
                    for i in range(cv):
                        best_scores_rounds.append(
                            rounds_df[f"split{i}_test_score"].iloc[0]
                        )

                fig.add_trace(
                    go.Box(
                        y=best_scores_rounds,
                        name="Best scores for every round",
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=1,
                    col=1,
                )

                # Second subplot: Best and Worst Trials
                for round in data_full_outer["round"].unique():
                    temp_df = data_full_outer[data_full_outer["round"] == round]
                    round_scores = []
                    for idx, row in temp_df.iterrows():
                        for i in range(cv):
                            round_scores.append(row[f"split{i}_test_score"])
                            if row["ranked"] == 1:
                                best_cv.append(row[f"split{i}_test_score"])

                    fig.add_trace(
                        go.Box(
                            y=round_scores,
                            name=f"Round {round+1}",
                            boxpoints="all",
                            jitter=0.3,
                            pointpos=-1.8,
                        ),
                        row=1,
                        col=2,
                    )

                fig.add_trace(
                    go.Box(
                        name="Best trial of all rounds",
                        y=best_cv,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=1,
                    col=2,
                )

                # Update layout for better readability
                fig.update_layout(
                    title=f"Model Evaluation Results With {evaluation} Method",
                    height=600,
                    width=1200,
                    showlegend=False,
                )

                fig.update_yaxes(title_text="Score", row=1, col=1)
                fig.update_yaxes(title_text="Score", row=1, col=2)

        fig.show()

        if calculate_shap:
            self.shap_values = shaps_array
            return self.best_estimator, data_full_outer, shaps_array
        else:
            return self.best_estimator, data_full_outer

    @scoring_check
    def bootstrap_validation(
        self, scoring="matthews_corrcoef", n_iter=100, test_size=0.2, boxplot=True
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=test_size
        )

        bootstrap_scores = []
        _estimator = self.best_estimator
        for i in tqdm(range(n_iter), desc="Bootstrap validation"):
            X_train_res, y_train_res = resample(X_train, y_train, random_state=i)
            _estimator.fit(X_train_res, y_train_res)
            y_pred = _estimator.predict(X_test)
            score = metrics.get_scorer(scoring)._score_func(y_test, y_pred)
            bootstrap_scores.append(score)
        return bootstrap_scores

    def c_v(self):
        cv = self.params["cv"]
        n_trials = self.params["n_trials"]
        direction = self.params["direction"]
        estimator_name = self.params["estimator_name"]
        calculate_shap = self.params["calculate_shap"]
        X = self.X
        y = self.y
        param_grid = self.param_grid
        scorer = self.params["scorer"]
        evaluation = self.params["evaluation"]
        rounds = self.params["rounds"]
        local_data_full_outer = pd.DataFrame()

        if calculate_shap:
            x_shap = np.zeros((X.shape[0], X.shape[1]))

        if cv < 2:
            raise ValueError("Cross-validation rounds must be greater than 1")

        list_train_test_indices = []
        list_x_train = []
        list_x_test = []
        list_y_train = []
        list_y_test = []
        scores = []
        scores_per_cv = []

        if evaluation == "cv_simple" or evaluation == "bootstrap":
            rounds = 1
        elif evaluation == "cv_rounds":
            rounds = rounds
        else:
            raise ValueError(
                "Evaluation method must be either 'cv_simple' or 'cv_rounds'"
            )

        # split the train and test sets for cv and rounds cv evaluations
        for i in range(rounds):
            cv_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)
            temp_train_test_indices = list(cv_splits.split(X, y))
            list_train_test_indices.append(temp_train_test_indices)
            for train_index, test_index in temp_train_test_indices:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                list_x_train.append(X_train)
                list_x_test.append(X_test)
                list_y_train.append(y_train)
                list_y_test.append(y_test)

        # train model
        cv_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)

        def objective(trial):
            try:
                clf = param_grid[estimator_name](trial)
                for set in range(cv):
                    clf.fit(list_x_train[set], list_y_train[set])
                    scores.append(scorer(clf, list_x_test[set], list_y_test[set]))
                score = np.mean(scores)
                return score
            except Exception as e:
                print(
                    f"{e}.\nThe None Score and the Hyperparameters of it will not be saved."
                )
                return None

        study = optuna.create_study(sampler=TPESampler(), direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
        best_params = study.best_params
        best_model = self.available_clfs[estimator_name]
        best_model.set_params(**best_params)
        best_model.fit(X, y)
        self.best_estimator = best_model

        scores_per_cv = []
        for i in range(rounds):
            scores = []
            for train_index, test_index in list_train_test_indices[i]:
                temp_model = copy.deepcopy(best_model)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                temp_model.fit(X_train, y_train)
                scores.append(scorer(temp_model, X_test, y_test))

                if calculate_shap:
                    shap_values = self.calc_shap(X_train, best_model)
                    x_shap[train_index, :] = np.add(
                        x_shap[train_index, :], shap_values.values
                    )

            scores_per_cv.append(scores)

        if evaluation == "cv_rounds" or evaluation == "cv_simple":
            all_params = [trial.params for trial in study.trials]
            trial_ids = [trial.number for trial in study.trials]
            for params, trial in zip(all_params, trial_ids):
                for round_num in range(rounds):
                    row = {}
                    for cv_trial in range(cv):
                        row_key = f"split{cv_trial}_test_score"
                        row[row_key] = scores_per_cv[round_num][cv_trial]

                    row["mean_test_score"] = np.mean(scores_per_cv[round_num])
                    row["std_test_score"] = np.std(scores_per_cv[round_num])

                    valid_scores = [
                        score for score in scores_per_cv[round_num] if score is not None
                    ]
                    if valid_scores:
                        sem = np.std(valid_scores) / np.sqrt(len(valid_scores))
                    else:
                        sem = 0

                    row["sem_test_score"] = sem
                    row["params"] = params
                    row["trial"] = trial
                    row["round"] = round_num

                    row_df = pd.DataFrame([row])
                    local_data_full_outer = pd.concat(
                        [local_data_full_outer, row_df], axis=0
                    )

            local_data_full_outer.reset_index(drop=True, inplace=True)

            local_data_full_outer = local_data_full_outer.sort_values(
                "mean_test_score", ascending=False
            )
            local_data_full_outer.reset_index(drop=True, inplace=True)
            local_data_full_outer["ranked"] = local_data_full_outer.index + 1

        elif evaluation == "bootstrap":
            bootstrap_scores = self.bootstrap_validation(
                scoring=self.scoring, n_iter=100, test_size=0.2
            )
            local_data_full_outer["Scores"] = bootstrap_scores
            local_data_full_outer["mean_test_score"] = np.mean(bootstrap_scores)
            local_data_full_outer["std_test_score"] = np.std(bootstrap_scores)
            valid_scores = [score for score in bootstrap_scores if score is not None]
            if valid_scores:
                sem = np.std(valid_scores) / np.sqrt(len(valid_scores))
            else:
                sem = 0
            local_data_full_outer["sem_test_score"] = sem
            local_data_full_outer["params"] = best_params
            local_data_full_outer["round"] = "bootstrap"
            local_data_full_outer["ranked"] = 1

        if calculate_shap:
            x_shap = x_shap / (rounds * (cv - 1))
            return best_model, local_data_full_outer, x_shap
        else:
            return best_model, local_data_full_outer
