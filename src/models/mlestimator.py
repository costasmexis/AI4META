import logging
import os
import numpy as np
import optuna
import pandas as pd
import sklearn.metrics as metrics
from src.data_manipulation import DataLoader
from optuna.samplers import TPESampler
from plotly.subplots import make_subplots

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
import warnings

from src.features.features_selection import _preprocess
from src.data_manipulation.class_balance import _class_balance
from src.utils.parameters_grid import optuna_grid
from src.utils.translators import AVAILABLE_CLFS
from src.utils.validation import _validation
from src.utils.model_manipulation.model_instances import _create_model_instance
from src.utils.model_manipulation.inner_selection import _one_sem_model, _gso_model
from src.utils.model_evaluation.evaluation import _evaluate
from src.utils.plots import _plot_per_metric
from src.database.database_input import insert_to_db

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

    def search_cv(
        self,
        search_type="bayesian_search",
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
        extra_metrics = ['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','average_precision','specificity','matthews_corrcoef'],
        warnings_filter = False,
        info_to_db = False,
        class_balance = None,
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
        if search_type not in ["random_search", "grid_search", "bayesian_search"]:
            raise ValueError(
                f"Invalid search type: {search_type}. Select one of the following: ['random_search', 'bayesian_search']"
            )
        self.model_selection_way = search_type

        self.config_cv = locals()
        self.config_cv.pop("self", None)
        self.config_cv = _validation(self.config_cv, self.model_selection_way, self.X, self.csv_dir, self.label, self.available_clfs)
        #TODO: _add if for the search type in the parameters check
        
        X, _, num_feature = _preprocess(self.X, self.y, self.config_cv['num_features'], self.config_cv, features_names_list = features_names_list)

        if param_grid is None:
            if search_type == "bayesian_search":
                self.param_grid = optuna_grid["NestedCV"]
            else:
                self.param_grid = optuna_grid["SklearnParameterGrid"]
            print("Using default parameter grid")
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid
            
        if warnings_filter:
            warnings.filterwarnings("ignore")

        custom_cv_splits = StratifiedKFold(n_splits=splits, shuffle=True).split(X, self.y)
        
        processors_available = os.cpu_count()
        if processors == -1:
            pass
        elif processors > processors_available:
            print(f"Warning: {processors} processors are not available. Using {processors_available} processors instead.")
            processors = processors_available
        elif processors < 1:
            print(f"Warning: {processors} processors are not available. Using 1 processor instead.")
            processors = 1
        
        if search_type in ['random_search', 'grid_search']:
            if search_type == "random_search":
                search_cv = RandomizedSearchCV(
                    estimator = AVAILABLE_CLFS[estimator_name],
                    param_distributions = self.param_grid[estimator_name],
                    scoring=scoring,
                    cv=custom_cv_splits,
                    n_iter=n_trials,
                    n_jobs=processors,
                )
            else:
                search_cv = GridSearchCV(
                    estimator = AVAILABLE_CLFS[estimator_name],
                    param_grid = self.param_grid[estimator_name],
                    scoring=scoring,
                    cv=custom_cv_splits,
                    n_jobs=processors,
                )
                
            search_cv.fit(X, self.y)
            best_params = search_cv.best_params_
            
        else:
            search_cv = optuna.integration.OptunaSearchCV(
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
            
            search_cv.fit(X, self.y)
            model_trials = search_cv.trials_

            if (inner_selection == "one_sem") or (inner_selection == "one_sem_grd"):
                samples = X.shape[0]
                # Find simpler parameters with the one_sem method if there are any
                simple_model_params = _one_sem_model(model_trials, estimator_name, samples, splits, inner_selection)
            elif (inner_selection == "gso_1") or (inner_selection == "gso_2"):
                # Find parameters with the smaller gap score with gso_1 method if there are any
                simple_model_params = _gso_model(model_trials, self.name, splits, inner_selection)
            else:
                simple_model_params = search_cv.best_params_
            
            best_params = simple_model_params
            
        self.config_cv['hyperparameters'] = best_params
        best_model = _create_model_instance(estimator_name, best_params)
        self.best_estimator = best_model

        scores_df, shaps_array = _evaluate(X, self.y, best_model, best_params, self.config_cv)
    
        if boxplot:
            _plot_per_metric(scores_df, estimator_name, self.config_cv['inner_selection'], evaluation)

        if info_to_db:
            scores_db_df = pd.DataFrame({col: [scores_df[col].tolist()] for col in scores_df.columns})
            insert_to_db(scores_db_df, self.config_cv, self.database_name)

        if calculate_shap:
            self.shap_values = shaps_array

        return scores_df
