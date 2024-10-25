import copy
import logging
import os

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import shap
import sklearn
import sklearn.metrics as metrics
from catboost import CatBoostClassifier
from dataloader import DataLoader
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.metrics import make_scorer


from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
    StratifiedShuffleSplit
)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from tqdm import tqdm
from xgboost import XGBClassifier
import warnings

from .optuna_grid import optuna_grid


def scoring_check(scoring: str) -> None:
    if scoring not in sklearn.metrics.get_scorer_names():
        raise ValueError(
            f"Invalid scoring metric: {scoring}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
        )


class MachineLearningEstimator(DataLoader):
    def __init__(self, label, csv_dir, estimator=None, param_grid=None):
        super().__init__(label, csv_dir)
        self.estimator = estimator
        self.name = estimator.__class__.__name__
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        self.scoring = None
        self.model_selection_way = None

        self.available_clfs = {
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
            "LogisticRegression": LogisticRegression(),
            "ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga"),
            "XGBClassifier": XGBClassifier(),
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "SVC": SVC(),
            "LGBMClassifier": LGBMClassifier(),
            "GaussianProcessClassifier": GaussianProcessClassifier(),
            "CatBoostClassifier": CatBoostClassifier(),
        }

        # Checking if estimator is valid
        if self.estimator is not None:
            if isinstance(estimator, str):
                self.name = estimator

            if self.name not in self.available_clfs.keys():
                raise ValueError(
                    f"Invalid estimator: {self.name}. Select one of the following: {list(self.available_clfs.keys())}"
                )
        elif self.estimator is None:
            print("There is no selected classifier.")

    def _set_optuna_verbosity(self, level):
        """ Adjust Optuna's verbosity level """
        optuna.logging.set_verbosity(level)
        logging.getLogger("optuna").setLevel(level)

    def _preprocess(
            self,
            X,
            y,
            scoring,
            features_names_list,
            feat_num,
            feat_way,
            estimator_name,
            normalization,
            missing_values,
        ):
        """Preprocess the data before fitting the estimator.

        This method performs various preprocessing steps on the input data
        based on the provided parameters. It handles missing values, feature
        selection, normalization, and scoring.

        :param X: The input features.
        :type X: array-like
        :param y: The target variable.
        :type y: array-like
        :param scoring: The scoring metric to use for evaluation.
        :type scoring: str
        :param features_names_list: The list of feature names to select.
        :type features_names_list: list or None
        :param feat_num: The number of features to select.
        :type feat_num: int or None
        :param feat_way: The feature selection method.
        :type feat_way: str or None
        :param estimator_name: The name of the estimator to use.
        :type estimator_name: str or None
        :param normalization: The normalization method to use.
        :type normalization: str or None
        :param missing_values: The method to handle missing values.
        :type missing_values: str or None
        :return: The preprocessed features, target variable, and estimator.
        :rtype: tuple
        """
        scoring_check(scoring)
        # scoring_function = getattr(metrics, scoring)
        scorer = metrics.get_scorer(scoring)

        # print(f"Scoring function: {scoring_function}")
        # scorer = make_scorer(scoring_function)

        self.scoring = scorer

        X = X or self.X
        y = y or self.y

        if missing_values is not None:
            X = self.missing_values(X, method=missing_values)

        if normalization is not None:
            X = self.normalize(X, method=normalization)

        if features_names_list is not None:
            X = X[features_names_list]

        elif feat_num is not None:
            selected = self.feature_selection(X, y, feat_way, feat_num)
            X = X[selected]

        if estimator_name is None:
            estimator_name = self.name
        else:
            estimator_name = estimator_name

        estimator = self.available_clfs[estimator_name]
        return X, y, estimator

    def grid_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        features_names_list=None,
        cv=5,
        estimator_name=None,
        evaluation="cv_simple",
        rounds=10,
        feat_num=None,
        feat_way="mrmr",
        missing_values="median",
        calculate_shap=False,
        param_grid=None,
        normalization="minmax",
        boxplot=True,
        extra_metrics = ['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','matthews_corrcoef'],
        warnings_filter=False,
        processors = -1
    ):
        """
        Perform grid search for hyperparameter tuning.

        :param X: The input features, defaults to None
        :type X: array-like, optional
        :param y: The target variable, defaults to None
        :type y: array-like, optional
        :param scoring: The scoring metric to optimize, defaults to "matthews_corrcoef"
        :type scoring: str, optional
        :param features_names_list: The list of feature names, defaults to None
        :type features_names_list: list, optional
        :param cv: The number of cross-validation folds, defaults to 5
        :type cv: int, optional
        :param estimator_name: The name of the estimator, defaults to None
        :type estimator_name: str, optional
        :param evaluation: The evaluation method, defaults to "cv_simple"
        :type evaluation: str, optional
        :param rounds: The number of evaluation rounds, defaults to 10
        :type rounds: int, optional
        :param feat_num: The number of features to select, defaults to None
        :type feat_num: int, optional
        :param feat_way: The feature selection method, defaults to "mrmr"
        :type feat_way: str, optional
        :param missing_values: The method to handle missing values, defaults to "median"
        :type missing_values: str, optional
        :param calculate_shap: Whether to calculate SHAP values, defaults to False
        :type calculate_shap: bool, optional
        :param param_grid: The grid of hyperparameters to search over, defaults to None
        :type param_grid: dict, optional
        :param normalization: The method of feature normalization, defaults to "minmax"
        :type normalization: str, optional
        :param boxplot: Whether to plot evaluation results as boxplots, defaults to True
        :type boxplot: bool, optional
        :return: The best estimator and evaluation results
        :rtype: tuple
        """
        # scoring_check(scoring)
        self.model_selection_way = "grid_search"

        X, y, estimator = self._preprocess(
            X,
            y,
            scoring,
            features_names_list,
            feat_num,
            feat_way,
            estimator_name,
            normalization,
            missing_values
        )

        if param_grid is None:
            self.param_grid = optuna_grid["SklearnParameterGrid"]
            print("Using default parameter grid")
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid
        
        if warnings_filter:
            warnings.filterwarnings("ignore")

        custom_cv_splits = self._splitter(X, y, cv, evaluation, rounds)
        
        processors_available = os.cpu_count()
        if processors == -1:
            pass
        elif processors > processors_available:
            print(f"Warning: {processors} processors are not available. Using {processors_available} processors instead.")
            processors = processors_available
        elif processors < 1:
            print(f"Warning: {processors} processors are not available. Using 1 processor instead.")
            processors = 1

        random_search = GridSearchCV(
            estimator,
            self.param_grid[estimator_name],
            scoring=scoring,
            cv=custom_cv_splits,
            n_jobs=processors,
        )
        random_search.fit(X, y)
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_

        print(f"Estimator: {estimator_name}")
        print(f"Best parameters: {best_params}")
        print(f"Best {scoring}: {best_score}")

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
            
        

        if calculate_shap:
            _, eval_df, shaps_array = self.evaluate(
                X,
                y,
                cv,
                evaluation,
                rounds,
                self.best_estimator,
                best_params,
                random_search,
                calculate_shap,
                extra_metrics
            )
        else:
            _, eval_df = self.evaluate(
                X,
                y,
                cv,
                evaluation,
                rounds,
                self.best_estimator,
                best_params,
                random_search,
                calculate_shap,
                extra_metrics
            )

        if boxplot:
            self._eval_boxplot(estimator_name, eval_df, cv, evaluation)

        if calculate_shap:
            self.shap_values = shaps_array
            return self.best_estimator, eval_df, shaps_array
        else:
            return self.best_estimator, eval_df

    def random_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        features_names_list=None,
        rounds=10,
        cv=5,
        n_iter=100,
        estimator_name=None,
        evaluation="cv_simple",
        feat_num=None,
        feat_way="mrmr",
        missing_values="median",
        calculate_shap=False,
        param_grid=None,
        normalization="minmax",
        boxplot=True,
        extra_metrics = ['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','matthews_corrcoef'],
        warnings_filter=False,
        processors = -1
    ):
        """
        Perform a random search for hyperparameter optimization.

        :param X: The input features, defaults to None.
        :type X: array-like, optional
        :param y: The target variable, defaults to None.
        :type y: array-like, optional
        :param scoring: The scoring metric to optimize, defaults to "matthews_corrcoef".
        :type scoring: str, optional
        :param features_names_list: The list of feature names, defaults to None.
        :type features_names_list: list, optional
        :param rounds: The number of rounds to perform the random search, defaults to 10.
        :type rounds: int, optional
        :param cv: The number of cross-validation folds, defaults to 5.
        :type cv: int, optional
        :param n_iter: The number of parameter settings that are sampled, defaults to 100.
        :type n_iter: int, optional
        :param estimator_name: The name of the estimator, defaults to None.
        :type estimator_name: str, optional
        :param evaluation: The evaluation method, defaults to "cv_simple".
        :type evaluation: str, optional
        :param feat_num: The number of features to select, defaults to None.
        :type feat_num: int, optional
        :param feat_way: The feature selection method, defaults to "mrmr".
        :type feat_way: str, optional
        :param missing_values: The method to handle missing values, defaults to "median".
        :type missing_values: str, optional
        :param calculate_shap: Whether to calculate SHAP values, defaults to False.
        :type calculate_shap: bool, optional
        :param param_grid: The parameter grid for the estimator, defaults to None.
        :type param_grid: dict, optional
        :param normalization: The method for feature normalization, defaults to "minmax".
        :type normalization: str, optional
        :param boxplot: Whether to plot boxplots of evaluation results, defaults to True.
        :type boxplot: bool, optional
        :return: The best estimator and evaluation results and/or shap values.
        :rtype: tuple
        """

        self.model_selection_way = "random_search"

        X, y, estimator = self._preprocess(
            X,
            y,
            scoring,
            features_names_list,
            feat_num,
            feat_way,
            estimator_name,
            normalization,
            missing_values,
        )

        if param_grid is None:
            self.param_grid = optuna_grid["SklearnParameterGrid"]
            print("Using default parameter grid")
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid
            
        if warnings_filter:
            warnings.filterwarnings("ignore")

        custom_cv_splits = self._splitter(X, y, cv, evaluation, rounds)
        
        processors_available = os.cpu_count()
        if processors == -1:
            pass
        elif processors > processors_available:
            print(f"Warning: {processors} processors are not available. Using {processors_available} processors instead.")
            processors = processors_available
        elif processors < 1:
            print(f"Warning: {processors} processors are not available. Using 1 processor instead.")
            processors = 1

        random_search = RandomizedSearchCV(
            estimator,
            self.param_grid[estimator_name],
            scoring=scoring,
            cv=custom_cv_splits,
            n_iter=n_iter,
            n_jobs=processors,
        )
        random_search.fit(X, y)
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_

        print(f"Estimator: {estimator_name}")
        print(f"Best parameters: {best_params}")
        print(f"Best {scoring}: {best_score}")

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

        if calculate_shap:
            _, eval_df, shaps_array = self.evaluate(
                X,
                y,
                cv,
                evaluation,
                rounds,
                self.best_estimator,
                best_params,
                random_search,
                calculate_shap,
                extra_metrics
            )
        else:
            _, eval_df = self.evaluate(
                X,
                y,
                cv,
                evaluation,
                rounds,
                self.best_estimator,
                best_params,
                random_search,
                calculate_shap,
                extra_metrics
            )

        if boxplot:
            self._eval_boxplot(estimator_name, eval_df, cv, evaluation)

        if calculate_shap:
            self.shap_values = shaps_array
            return self.best_estimator, eval_df, shaps_array
        else:
            return self.best_estimator, eval_df

    def bayesian_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        features_names_list=None,
        rounds=10,
        cv=5,
        direction="maximize",
        n_trials=100,
        estimator_name=None,
        evaluation="cv_rounds",
        feat_num=None,
        feat_way="mrmr",
        missing_values="median",
        boxplot=True,
        calculate_shap=False,
        param_grid=None,
        normalization="minmax",
        training_method = "validation_score",
        extra_metrics = ['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','matthews_corrcoef'],
        warnings_filter = False,
        processors = -1
    ):
        """
        Perform a Bayesian search for hyperparameter optimization.

        :param X: The input features, defaults to None.
        :type X: array-like, optional
        :param y: The target variable, defaults to None.
        :type y: array-like, optional
        :param scoring: The scoring metric to optimize, defaults to "matthews_corrcoef".
        :type scoring: str, optional
        :param features_names_list: The list of feature names, defaults to None.
        :type features_names_list: list, optional
        :param rounds: The number of rounds for cross-validation, defaults to 10.
        :type rounds: int, optional
        :param cv: The number of cross-validation folds, defaults to 5.
        :type cv: int, optional
        :param direction: The direction to optimize the scoring metric, defaults to "maximize".
        :type direction: str, optional
        :param n_trials: The number of trials for the Bayesian search, defaults to 100.
        :type n_trials: int, optional
        :param estimator_name: The name of the estimator, defaults to None.
        :type estimator_name: str, optional
        :param evaluation: The evaluation method, defaults to "cv_rounds".
        :type evaluation: str, optional
        :param feat_num: The number of features to select, defaults to None.
        :type feat_num: int, optional
        :param feat_way: The feature selection method, defaults to "mrmr".
        :type feat_way: str, optional
        :param missing_values: The method to handle missing values, defaults to "median".
        :type missing_values: str, optional
        :param boxplot: Whether to plot the evaluation results as a boxplot, defaults to True.
        :type boxplot: bool, optional
        :param calculate_shap: Whether to calculate SHAP values, defaults to False.
        :type calculate_shap: bool, optional
        :param param_grid: The hyperparameter grid, defaults to None.
        :type param_grid: dict, optional
        :param normalization: The method for feature normalization, defaults to "minmax".
        :type normalization: str, optional
        :return: The best estimator and evaluation results and/or shap values.
        :rtype: tuple

        """
        # scoring_check(scoring)
        self.model_selection_way = "bayesian_search"
        self._set_optuna_verbosity(logging.ERROR)

        if training_method not in ['validation_score','one_sem','gso_1','gso_2','one_sem_grd']:
            raise ValueError("Invalid training_method. Select one of the following: ['validation_score','one_sem','gso_1','gso_2','one_sem_grd']")
        if evaluation not in ["train_test", "cv_rounds", "oob", "bootstrap"]:
            raise ValueError("Invalid evaluation. Select one of the following: ['train_test', 'cv_rounds', 'oob', 'bootstrap']")

        X, y, _ = self._preprocess(
            X,
            y,
            scoring,
            features_names_list,
            feat_num,
            feat_way,
            estimator_name,
            normalization,
            missing_values,
        )

        if param_grid is None:
            self.param_grid = optuna_grid["NestedCV"]
            # self.param_grid = optuna_grid["ManualSearch"]
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid

        if warnings_filter:
            warnings.filterwarnings("ignore")

        processors_available = os.cpu_count()
        if processors == -1:
            pass
        elif processors > processors_available:
            print(f"Warning: {processors} processors are not available. Using {processors_available} processors instead.")
            processors = processors_available
        elif processors < 1:
            print(f"Warning: {processors} processors are not available. Using 1 processor instead.")
            processors = 1
        
        custom_cv_splits = StratifiedKFold(n_splits=cv, shuffle=True).split(X, y)

        
        clf = optuna.integration.OptunaSearchCV(
            estimator=self.available_clfs[estimator_name],
            scoring=scoring,
            param_distributions=self.param_grid[estimator_name] ,
            cv=custom_cv_splits,
            return_train_score=True,
            n_jobs=processors,
            verbose=0,
            n_trials=n_trials,
            study=optuna.create_study(direction=direction, sampler=TPESampler()),
            subsample=0.7*X.shape[0]*(cv-1)/cv,
        )
        
        clf.fit(X, y)
        study = clf.study_
        model_trials = clf.trials_

        if (training_method == "one_sem") or (training_method == "one_sem_grd"):
            samples = X.shape[0]
            # Find simpler parameters with the one_sem method if there are any
            simple_model_params = self._one_sem_model(model_trials, estimator_name, samples, cv, training_method)
        elif (training_method == "gso_1") or (training_method == "gso_2"):
            # Find parameters with the smaller gap score with gso_1 method if there are any
            simple_model_params = self._gso_model(model_trials, self.name, cv, training_method)
        else:
            simple_model_params = clf.best_params_
        
        best_params = simple_model_params
        
        best_model = self._create_model_instance(estimator_name, best_params)

        self.best_estimator = best_model
                    
        for trial in study.trials:
            # Check if the trial parameters match the selected parameters
            if trial.params == simple_model_params:
                matching_score = trial.value  # Extract the score of the matching trial
                break    
        
        best_score = clf.best_score_
        
        if estimator_name == 'ElasticNet':
            print(f"Estimator: LogisticRegression with ElasticNet penalty")
        else:
            print(f"Estimator: {estimator_name}")
        print(f"Best parameters: {best_params}")
        
        if training_method == "validation_score":
            print(f"Best trials score: {matching_score}.")
        else:
            if matching_score == best_score:
                print(f"Best trials score: {best_score}. Using {training_method} th best score is the same.")
            else:
                print(f"Best trials score wiuth validation method: {best_score}. Using {training_method} th best score is {matching_score}.")

        if (calculate_shap) and (evaluation == 'cv_rounds'):
            best_model, eval_df, shaps_array = self._evaluate(
                X,
                y,
                cv,
                evaluation,
                rounds,
                best_model,
                best_params,
                study,
                calculate_shap,
                extra_metrics,
                training_method,
                estimator_name,
                features_names_list,
                feat_num
            )
        else:
            best_model, eval_df = self._evaluate(
                X,
                y,
                cv,
                evaluation,
                rounds,
                best_model,
                best_params,
                study,
                calculate_shap,
                extra_metrics,
                training_method,
                estimator_name,
                features_names_list,
                feat_num
            )

        if boxplot:
            self._eval_boxplot(estimator_name, eval_df, cv, evaluation)

        if calculate_shap and (evaluation == 'cv_rounds'):
            self.shap_values = shaps_array
            return self.best_estimator, eval_df, shaps_array
        else:
            return self.best_estimator, eval_df

        
    def _calc_shap(self, X_train, X_test, model):
        try:
            explainer = shap.Explainer(model, X_train)
        except TypeError as e:
            if (
                "The passed model is not callable and cannot be analyzed directly with the given masker!"
                in str(e)
            ):
                print(
                    "Switching to predict_proba due to compatibility issue with the model."
                )
                explainer = shap.Explainer(lambda x: model.predict_proba(x), X_train)
            else:
                raise TypeError(e)
        try:
            if self.best_estimator.__class__.__name__ in ["LGBMClassifier", "CatBoostClassifier","RandomForestClassifier"]:
                shap_values = explainer(X_test, check_additivity=False)
            else: 
                shap_values = explainer(X_test)
        except ValueError:
            num_features = X_test.shape[1]
            max_evals = 2 * num_features + 1
            shap_values = explainer(X_test, max_evals=max_evals)

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        else:
            pass
        return shap_values

    def _bootstrap_validation(
            self, X, y, model, extra_metrics=None):#, calculate_shap=False
        """Performs bootstrap validation for model evaluation.
        :return: A tuple of (bootstrap_scores, extra_metrics_scores).
        :rtype: tuple
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True
        )

        bootstrap_scores = []
        extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}    
        
        for i in tqdm(range(100), desc="Bootstrap validation"):
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=0.25, shuffle=True, random_state=i
            #     )
            model_bootstrap = copy.deepcopy(model)
            X_train_res, y_train_res = resample(X_train, y_train, random_state=i)
            # model_bootstrap.fit(X_train_res, y_train_res)
            # y_pred = model_bootstrap.predict(X_test)
            model_bootstrap.fit(X_train_res, y_train_res)
            y_pred = model_bootstrap.predict(X_test)
            
            # Calculate the main scoring metric
            score = metrics.get_scorer(self.scoring)._score_func(y_test, y_pred)
            bootstrap_scores.append(score)
            
            # Calculate and store extra metrics
            if extra_metrics is not None:
                for extra in extra_metrics:
                    extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
                    extra_metrics_scores[extra].append(extra_score)
            
            # # Calculate and accumulate SHAP values
            # if calculate_shap:
            #     shap_values = self.calc_shap(X_train, X_test, model_bootstrap)
            #     all_shap_values[X_test.index] += shap_values.values
            #     counts[X_test.index] += 1
        
        # Calculate the mean SHAP values by dividing accumulated SHAP values by counts
        # if calculate_shap:
        #     mean_shap_values = all_shap_values / counts[:, None]        
        #     return bootstrap_scores, extra_metrics_scores, mean_shap_values
        # else:
        return bootstrap_scores, extra_metrics_scores
    
    def _oob_validation(
            self, X, y, model, extra_metrics=None
        ):

        oob_scores = []
        extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
        
        for i in tqdm(range(100), desc="OOB validation"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, shuffle=True
            )
            X_train_res, y_train_res = resample(X_train, y_train, random_state=i)
            model_oob = copy.deepcopy(model)
            model_oob = model_oob.fit(X_train_res, y_train_res)
            y_pred = model_oob.predict(X_test)
        
            score = metrics.get_scorer(self.scoring)._score_func(y_test, y_pred)
            oob_scores.append(score)
            
            # Calculate and store extra metrics
            if extra_metrics is not None:
                for extra in extra_metrics:
                    extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
                    extra_metrics_scores[extra].append(extra_score)

        return oob_scores, extra_metrics_scores

    def _train_test_validation(self, X, y, model, extra_metrics=None):
        tt_prop_scores = []
        extra_metrics_scores = {extra: [] for extra in extra_metrics} if extra_metrics else {}
        sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

        for i, (train_index, test_index) in tqdm(enumerate(sss.split(X, y)), desc="TT prop validation"):
            # Use .iloc for DataFrame X and NumPy indexing for array y
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Deepcopy the model and fit it on the train set
            model_tt_prop = copy.deepcopy(model)
            model_tt_prop = model_tt_prop.fit(X_train, y_train)  
            y_pred = model_tt_prop.predict(X_test)

            # Evaluate the primary metric
            score = metrics.get_scorer(self.scoring)._score_func(y_test, y_pred)
            tt_prop_scores.append(score)

            # Calculate and store extra metrics
            if extra_metrics is not None:
                for extra in extra_metrics:
                    extra_score = metrics.get_scorer(extra)._score_func(y_test, y_pred)
                    extra_metrics_scores[extra].append(extra_score)

        return tt_prop_scores, extra_metrics_scores

    def _eval_boxplot(self, estimator_name, eval_df, cv, evaluation):
        """
        Generate a boxplot to visualize the model evaluation results.
        
        :param estimator_name: The name of the estimator.
        :type estimator_name: str
        :param eval_df: The evaluation dataframe containing the scores.
        :type eval_df: pandas.DataFrame
        :param cv: The number of cross-validation folds.
        :type cv: int
        :param evaluation: The evaluation method to use ("bootstrap", "cv_simple", or any other custom method).
        :type evaluation: str
        """
        
        results_dir = "Final_Model_Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if (evaluation == "bootstrap") or (evaluation == "oob") or (evaluation == "train_test"):
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    y=eval_df["Scores"],
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
        # elif evaluation == "cv_simple":
        #     fig = go.Figure()
        #     all_scores = []
        #     best_cv_scores = []
        #     for i in range(cv):
        #         for row in range(eval_df.shape[0]):
        #             all_scores.append(eval_df[f"split{i}_test_score"].iloc[row])

        #     fig.add_trace(
        #         go.Box(
        #             y=all_scores,
        #             name="All trials scores",
        #             boxpoints="all",
        #             jitter=0.3,
        #             pointpos=-1.8,
        #             boxmean=True,
        #         )
        #     )

        #     temp_best_cv = eval_df[eval_df["ranked"] == 1]
        #     for i in range(cv):
        #         best_cv_scores.append(temp_best_cv[f"split{i}_test_score"].iloc[0])

        #     fig.add_trace(
        #         go.Box(
        #             y=best_cv_scores,
        #             name="Best trial Scores",
        #             boxpoints="all",
        #             jitter=0.3,
        #             pointpos=-1.8,
        #             boxmean=True,
        #         )
        #     )

        #     fig.update_layout(
        #         title=f"Model Evaluation Results With {evaluation} Method",
        #         yaxis_title="Score",
        #         template="plotly_white",
            # )
        elif evaluation == "cv_rounds":
            # fig = make_subplots(
            #     rows=1, cols=2, subplot_titles=("Summary Boxplot", "Rounds Scores")
            # )
            fig = go.Figure()
            all_scores = []
            best_scores_rounds = []
            best_cv = []
            for row in range(eval_df.shape[0]):
                for score in eval_df["Scores"].iloc[row]:
                    all_scores.append(score)

            fig.add_trace(
                go.Box(
                    y=all_scores,
                    name="All trial Scores",
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )#,
                # row=1,
                # col=1,
            )

            # # Second subplot: Best and Worst Trials
            # for round in eval_df["round"].unique():
            #     temp_df = eval_df[eval_df["round"] == round]
            #     round_scores = []
            #     for idx, row in temp_df.iterrows():
            #         for i in range(cv):
            #             round_scores.append(row[f"split{i}_test_score"])
            #             # if row["ranked"] == 1:
            #             #     best_cv.append(row[f"split{i}_test_score"])

            #     fig.add_trace(
            #         go.Box(
            #             y=round_scores,
            #             name=f"Round {round+1}",
            #             boxpoints="all",
            #             jitter=0.3,
            #             pointpos=-1.8,
            #         ),
            #         row=1,
            #         col=2,
            #     )
                
            # Update layout for better readability
            fig.update_layout(
                title=f"Model Evaluation Results With {evaluation} Method",
                height=600,
                width=1200,
                showlegend=False,
            )

            # fig.update_yaxes(title_text="Score", row=1, col=1)
            # fig.update_yaxes(title_text="Score", row=1, col=2)
        elif evaluation == "train_test":
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=eval_df["round"],  # You can change this to any x-axis variable you want
                    y=eval_df["Scores"],     # Y-axis variable
                    mode='markers',          # Use markers for scatter plot
                    name=estimator_name,
                )
            )

            fig.update_layout(
                title=f"Model Evaluation Results With {evaluation} Method",
                xaxis_title="Score type",  # Change this to the relevant label for the x-axis
                yaxis_title="Score",
                template="plotly_white",
            )
        

        fig.show()

        # Save the plot to 'Results/final_model_evaluation.png'
        save_path = os.path.join(results_dir, f"final_model_evaluation_for_{estimator_name}.png")
        fig.write_image(save_path)

    def _evaluate(
        self, X, y, cv, evaluation, rounds, best_model, best_params, way, calculate_shap, extra_metrics, training_method, estimator_name,  features_names_list, feat_num
    ):
        """
        Evaluate the performance of a machine learning model using cross-validation or bootstrap methods.

        :param X: The input features.
        :type X: pandas.DataFrame
        :param y: The target variable.
        :type y: pandas.Series
        :param cv: The number of cross-validation folds.
        :type cv: int
        :param evaluation: The evaluation method to use. Must be either 'cv_simple', 'bootstrap', or 'cv_rounds'.
        :type evaluation: str
        :param rounds: The number of rounds for cross-validation or bootstrap evaluation.
        :type rounds: int
        :param best_model: The best model obtained from model selection.
        :type best_model: object
        :param best_params: The best hyperparameters obtained from model selection.
        :type best_params: dict
        :param way: The model selection method used. Only required if evaluation is 'cv_rounds' or 'cv_simple'.
        :type way: object
        :param calculate_shap: Whether to calculate SHAP values or not.
        :type calculate_shap: bool
        :raises ValueError: If cv is less than 2.
        :raises ValueError: If evaluation method is not one of 'cv_simple', 'bootstrap', or 'cv_rounds'.
        :return: The best model, evaluation results, and SHAP values (if calculate_shap is True).
        :rtype: tuple
        """
        
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

        # if (evaluation == "oob") or (evaluation == "bootstrap") or (evaluation == "train_test"):
        #     rounds = 1
        # elif evaluation == "cv_rounds":
        #     rounds = rounds
        # else:
        #     raise ValueError(
        #         "Evaluation method must be either 'bootstrap, 'train_test', 'oob' or 'cv_rounds'"
        #     )

        if evaluation == "cv_rounds":# or evaluation == "cv_simple":
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

            scores_per_cv = []
            metric_lists = {}
            if extra_metrics is not None:
                for extra in extra_metrics:
                    metric_lists[extra] = []

            for i in range(rounds):
                scores = []
                for train_index, test_index in list_train_test_indices[i]:
                    # temp_model = copy.deepcopy(best_model)
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    best_model.fit(X_train, y_train)
                    scores.append(self.scoring(best_model, X_test, y_test))
                    
                    # Calculate and store scores for each extra metric
                    if extra_metrics is not None:
                        for extra in extra_metrics:
                            extra_score = metrics.get_scorer(extra)._score_func(y_test, best_model.predict(X_test))
                            metric_lists[extra].append(extra_score)

                    if calculate_shap:
                        shap_values = self.calc_shap(X_train, X_test, best_model)
                        x_shap[test_index, :] = np.add(
                            x_shap[test_index, :], shap_values.values
                        )
                    # del(temp_model)
                scores_per_cv.append(scores)
            
            if calculate_shap:
                x_shap = x_shap / (rounds)

            # if self.model_selection_way == "bayesian_search":
            #     all_params = [trial.params for trial in way.trials]
            #     # trial_ids = [trial.number for trial in way.trials]
            # elif self.model_selection_way == "random_search":
            #     all_params = way.cv_results_["params"]
            #     # trial_ids = list(range(len(all_params)))
            # else:
            #     all_params = way.cv_results_["params"]
            #     # trial_ids = list(range(len(all_params)))
            
            for round_num in range(rounds):
                row = {}
                scores = []
                for cv_trial in range(cv):
                    # row_key = f"split{cv_trial}_test_score"
                    # row[row_key] = scores_per_cv[round_num][cv_trial]
                    scores.append(scores_per_cv[round_num][cv_trial])
                    
                row['Scores'] = scores 
                    
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
                row["params"] = best_params
                row["round"] = "round_cv"
                row['train_mthd'] = training_method
                row['estimator'] = estimator_name
                if features_names_list is not None:
                    row['features'] = len(features_names_list)
                elif feat_num is not None:
                    row['features'] = feat_num
                else:
                    row['features'] = 'all'

                # Calculate and add extra metrics to the row
                if extra_metrics is not None:
                    for extra in extra_metrics:
                        extra_metric_scores = metric_lists[extra][round_num * cv:(round_num + 1) * cv]
                        row[f"{extra}"] = extra_metric_scores

                row_df = pd.DataFrame([row])
                local_data_full_outer = pd.concat(
                    [local_data_full_outer, row_df], axis=0
                )

            local_data_full_outer.reset_index(drop=True, inplace=True)
        
            if calculate_shap:
                return best_model, local_data_full_outer, x_shap

        elif (evaluation == "bootstrap") or (evaluation == "oob"):
            # if calculate_shap:
            #     bootstrap_scores, extra_metrics_scores, x_shap = self.bootstrap_validation(X, y, best_model, extra_metrics, calculate_shap=True)
            # else:
            if evaluation == "bootstrap":
                bootstrap_scores, extra_metrics_scores = self._bootstrap_validation(X, y, best_model, extra_metrics)#, calculate_shap=False)
            else:
                bootstrap_scores, extra_metrics_scores = self._oob_validation(X, y, best_model, extra_metrics)#, calculate_shap=False)
            local_data_full_outer["Scores"] = bootstrap_scores
            local_data_full_outer["mean_test_score"] = np.mean(bootstrap_scores)
            local_data_full_outer["std_test_score"] = np.std(bootstrap_scores)
            
            valid_scores = [score for score in bootstrap_scores if score is not None]
            if valid_scores:
                sem = np.std(valid_scores) / np.sqrt(len(valid_scores))
            else:
                sem = 0
            local_data_full_outer["sem_test_score"] = sem
            # print(best_params, best_model.get_params())
            local_data_full_outer["params"] =  local_data_full_outer.apply(lambda row: best_params.copy(), axis=1)
            if evaluation == "oob":
                local_data_full_outer["round"] = "oob"
            else:
                local_data_full_outer["round"] = "bootstrap"
            local_data_full_outer['train_mthd'] = training_method
            local_data_full_outer['estimator'] = estimator_name
            if features_names_list is not None:
                local_data_full_outer['features'] = len(features_names_list)
            elif feat_num is not None:
                local_data_full_outer['features'] = feat_num
            else:
                local_data_full_outer['features'] = 'all'

            # Calculate and add extra metrics for bootstrap validation
            if extra_metrics is not None:
                for extra in extra_metrics:
                    extra_metric_scores = extra_metrics_scores[extra]
                    local_data_full_outer[f"{extra}"] = extra_metric_scores
        
        elif evaluation == "train_test":
            tt_scores, extra_metrics_scores = self._train_test_validation(X, y, best_model, extra_metrics)
            
            local_data_full_outer["Scores"] = tt_scores
            local_data_full_outer["mean_test_score"] = np.mean(tt_scores)
            local_data_full_outer["std_test_score"] = np.std(tt_scores)
            
            valid_scores = [score for score in tt_scores if score is not None]
            if valid_scores:
                sem = np.std(valid_scores) / np.sqrt(len(valid_scores))
            else:
                sem = 0
            local_data_full_outer["sem_test_score"] = sem
            local_data_full_outer["params"] =  local_data_full_outer.apply(lambda row: best_params.copy(), axis=1)
            local_data_full_outer["round"] = "train_test"
            local_data_full_outer['train_mthd'] = training_method
            local_data_full_outer['estimator'] = estimator_name
            if features_names_list is not None:
                local_data_full_outer['features'] = len(features_names_list)
            elif feat_num is not None:
                local_data_full_outer['features'] = feat_num
            else:
                local_data_full_outer['features'] = 'all'

            # Calculate and add extra metrics for train_test validation
            if extra_metrics is not None:
                for extra in extra_metrics:
                    extra_metric_scores = extra_metrics_scores[extra]
                    local_data_full_outer[f"{extra}"] = extra_metric_scores

        return best_model, local_data_full_outer