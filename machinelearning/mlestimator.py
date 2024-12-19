import logging
import os
import numpy as np
import optuna
import pandas as pd
import sklearn.metrics as metrics
from dataloader import DataLoader
from optuna.samplers import TPESampler
from plotly.subplots import make_subplots

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
import warnings

from machinelearning.utils.optuna_grid import optuna_grid
from machinelearning.utils.translators import AVAILABLE_CLFS
from machinelearning.utils.calc_fnc import _parameters_check
from machinelearning.utils.filter_ftrs import _preprocess
from machinelearning.utils.balance_fnc import _class_balance
from machinelearning.utils.modinst_fnc import _create_model_instance
from machinelearning.utils.inner_selection_fnc import _one_sem_model, _gso_model
from machinelearning.utils.mle_utils.evaluation import _evaluate
from machinelearning.utils.plots_fnc import _plot_per_metric
from machinelearning.utils.mle_utils.database_fnc import _save_to_db

class MachineLearningEstimator(DataLoader):
    def __init__(self, label, csv_dir, database_name=None, estimator=None, param_grid=None):
        super().__init__(label, csv_dir)
        self.database_name = database_name
        self.estimator = estimator
        self.name = estimator.__class__.__name__
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        self.scoring = None
        self.model_selection_way = None
        self.available_clfs = AVAILABLE_CLFS
        self.shap_values = None

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

    # def _preprocess(
    #         self,
    #         X,
    #         y,
    #         scoring,
    #         features_names_list,
    #         num_features,
    #         feature_selection_type,
    #         estimator_name,
    #         normalization,
    #         missing_values_method,
    #     ):
    #     """Preprocess the data before fitting the estimator.

    #     This method performs various preprocessing steps on the input data
    #     based on the provided parameters. It handles missing values, feature
    #     selection, normalization, and scoring.

    #     :param X: The input features.
    #     :type X: array-like
    #     :param y: The target variable.
    #     :type y: array-like
    #     :param scoring: The scoring metric to use for evaluation.
    #     :type scoring: str
    #     :param features_names_list: The list of feature names to select.
    #     :type features_names_list: list or None
    #     :param num_features: The number of features to select.
    #     :type num_features: int or None
    #     :param feature_selection_type: The feature selection method.
    #     :type feature_selection_type: str or None
    #     :param estimator_name: The name of the estimator to use.
    #     :type estimator_name: str or None
    #     :param normalization: The normalization method to use.
    #     :type normalization: str or None
    #     :param missing_values_method: The method to handle missing values.
    #     :type missing_values_method: str or None
    #     :return: The preprocessed features, target variable, and estimator.
    #     :rtype: tuple
    #     """
    #     scoring_check(scoring)
    #     # scoring_function = getattr(metrics, scoring)
    #     scorer = metrics.get_scorer(scoring)

    #     self.scoring = scorer

    #     X = X or self.X
    #     y = y or self.y

    #     if missing_values_method is not None:
    #         X = self.missing_values_method(X, method=missing_values_method)

    #     if normalization is not None:
    #         X = self.normalize(X, method=normalization)

    #     if features_names_list is not None:
    #         X = X[features_names_list]

    #     elif num_features is not None:
    #         selected = self.feature_selection(X, y, feature_selection_type, num_features)
    #         X = X[selected]

    #     if estimator_name is None:
    #         estimator_name = self.name
    #     else:
    #         estimator_name = estimator_name

    #     estimator = self.available_clfs[estimator_name]
    #     return X, y, estimator

    def grid_search(
        self,
        X=None,
        y=None,
        scoring="matthews_corrcoef",
        features_names_list=None,
        splits=5,
        estimator_name=None,
        evaluation="cv_simple",
        rounds=10,
        num_features=None,
        feature_selection_type="mrmr",
        missing_values_method="median",
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
        :param num_features: The number of features to select, defaults to None
        :type num_features: int, optional
        :param feature_selection_type: The feature selection method, defaults to "mrmr"
        :type feature_selection_type: str, optional
        :param missing_values_method: The method to handle missing values, defaults to "median"
        :type missing_values_method: str, optional
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
            num_features,
            feature_selection_type,
            estimator_name,
            normalization,
            missing_values_method
        )

        if param_grid is None:
            self.param_grid = optuna_grid["SklearnParameterGrid"]
            print("Using default parameter grid")
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid
        
        if warnings_filter:
            warnings.filterwarnings("ignore")

        custom_cv_splits = self._splitter(X, y, splits, evaluation, rounds)
        
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
            splits=custom_cv_splits,
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

        for i in range(splits):
            usefull_cols.append(f"split{i}_test_score")
            
        

        if calculate_shap:
            _, eval_df, shaps_array = self.evaluate(
                X,
                y,
                splits,
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
                splits,
                evaluation,
                rounds,
                self.best_estimator,
                best_params,
                random_search,
                calculate_shap,
                extra_metrics
            )

        if boxplot:
            self._eval_boxplot(estimator_name, eval_df, splits, evaluation)

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
        splits=5,
        n_iter=100,
        estimator_name=None,
        evaluation="cv_simple",
        num_features=None,
        feature_selection_type="mrmr",
        missing_values_method="median",
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
        :param num_features: The number of features to select, defaults to None.
        :type num_features: int, optional
        :param feature_selection_type: The feature selection method, defaults to "mrmr".
        :type feature_selection_type: str, optional
        :param missing_values_method: The method to handle missing values, defaults to "median".
        :type missing_values_method: str, optional
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
            num_features,
            feature_selection_type,
            estimator_name,
            normalization,
            missing_values_method,
        )

        if param_grid is None:
            self.param_grid = optuna_grid["SklearnParameterGrid"]
            print("Using default parameter grid")
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid
            
        if warnings_filter:
            warnings.filterwarnings("ignore")

        custom_cv_splits = self._splitter(X, y, splits, evaluation, rounds)
        
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
            splits=custom_cv_splits,
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

        for i in range(splits):
            usefull_cols.append(f"split{i}_test_score")

        if calculate_shap:
            _, eval_df, shaps_array = self.evaluate(
                X,
                y,
                splits,
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
                splits,
                evaluation,
                rounds,
                self.best_estimator,
                best_params,
                random_search,
                calculate_shap,
                extra_metrics
            )

        if boxplot:
            self._eval_boxplot(estimator_name, eval_df, splits, evaluation)

        if calculate_shap:
            self.shap_values = shaps_array
            return self.best_estimator, eval_df, shaps_array
        else:
            return self.best_estimator, eval_df

    def bayesian_search(
        self,
        scoring="matthews_corrcoef",
        features_names_list=None,
        rounds=10,
        splits=5,
        direction="maximize",
        n_trials=100,
        estimator_name=None,
        evaluation="cv_rounds",
        num_features=None,
        feature_selection_type="mrmr",
        feature_selection_method="chi2",
        missing_values_method="median",
        boxplot=True,
        calculate_shap=False,
        param_grid=None,
        normalization="minmax",
        inner_selection = "validation_score",
        extra_metrics = ['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','matthews_corrcoef'],
        warnings_filter = False,
        info_to_db = False,
        class_balance = None,
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
        :param num_features: The number of features to select, defaults to None.
        :type num_features: int, optional
        :param feature_selection_type: The feature selection method, defaults to "mrmr".
        :type feature_selection_type: str, optional
        :param missing_values_method: The method to handle missing values, defaults to "median".
        :type missing_values_method: str, optional
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
        self.model_selection_way = "bayesian_search"
        self._set_optuna_verbosity(logging.ERROR)

        self.config_cv = locals()
        self.config_cv.pop("self", None)
        self.config_cv = _parameters_check(self.config_cv, self.model_selection_way, self.X, self.csv_dir, self.label, self.available_clfs)

        X, _, num_feature = _preprocess(self.X, self.y, self.config_cv['num_features'], self.config_cv, features_names_list = features_names_list)

        if param_grid is None:
            self.param_grid = optuna_grid["NestedCV"]
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
        
        custom_cv_splits = StratifiedKFold(n_splits=splits, shuffle=True).split(X, self.y)

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
            subsample=0.7*X.shape[0]*(splits-1)/splits,
        )
        
        clf.fit(X, self.y)
        model_trials = clf.trials_

        if (inner_selection == "one_sem") or (inner_selection == "one_sem_grd"):
            samples = X.shape[0]
            # Find simpler parameters with the one_sem method if there are any
            simple_model_params = _one_sem_model(model_trials, estimator_name, samples, splits, inner_selection)
        elif (inner_selection == "gso_1") or (inner_selection == "gso_2"):
            # Find parameters with the smaller gap score with gso_1 method if there are any
            simple_model_params = _gso_model(model_trials, self.name, splits, inner_selection)
        else:
            simple_model_params = clf.best_params_
        
        # Initiate the best model
        best_params = simple_model_params
        self.config_cv['hyperparameters'] = best_params
        best_model = _create_model_instance(estimator_name, best_params)
        self.best_estimator = best_model

        scores_df, shaps_array = _evaluate(X, self.y, best_model, best_params, self.config_cv)

        if boxplot:
            _plot_per_metric(scores_df, estimator_name, inner_selection, evaluation)
        
        if info_to_db:
            print('INFO TO DB')
            print(self.config_cv)
            print(scores_df.columns)
            _save_to_db(scores_df, self.config_cv)

        if calculate_shap and (evaluation == 'cv_rounds'):
            self.shap_values = shaps_array

        print('Model created and it is selected with the best parameters')
        return scores_df
        






    