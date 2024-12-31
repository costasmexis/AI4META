import logging
import os
import numpy as np
import optuna
import pandas as pd
import sklearn.metrics as metrics
from src.data.dataloader import DataLoader
from optuna.samplers import TPESampler

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
import warnings

from src.features.features_selection import _preprocess
from src.data.class_balance import _class_balance
from src.constants.parameters_grid import optuna_grid
from src.constants.translators import AVAILABLE_CLFS
from src.utils.validation.validation import _validation
from src.utils.model_manipulation.model_instances import _create_model_instance
from src.utils.model_manipulation.inner_selection import _one_sem_model, _gso_model
from src.utils.model_evaluation.evaluation import _evaluate
from src.utils.plots.plots import _plot_per_metric
from src.db.input import insert_to_db

class MachineLearningEstimator(DataLoader):
    def __init__(self, label, csv_dir, database_name=None, estimator=None, param_grid=None):
        super().__init__(label, csv_dir)
        self.database_name = database_name
        self.estimator = estimator
        self.name = estimator.__class__.__name__ if estimator else None
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        self.scoring = None
        self.model_selection_way = None
        self.available_clfs = AVAILABLE_CLFS
        self.shap_values = None

        if self.estimator and self.name not in self.available_clfs:
            raise ValueError(
                f"Invalid estimator: {self.name}. Available classifiers: {list(self.available_clfs.keys())}"
            )

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
        inner_selection="validation_score",
        extra_metrics=[
            'roc_auc', 'accuracy', 'balanced_accuracy', 'recall', 'precision',
            'f1', 'average_precision', 'specificity', 'matthews_corrcoef'
        ],
        warnings_filter=False,
        info_to_db=False,
        class_balance=None,
        processors=-1
    ):
        """
        Perform hyperparameter optimization using specified search methods.

        Parameters:
        -----------
        search_type : str, optional
            Type of search ('random_search', 'grid_search', 'bayesian_search'). Defaults to 'bayesian_search'.
        scoring : str, optional
            Metric to optimize. Defaults to 'matthews_corrcoef'.
        features_names_list : list, optional
            Feature names to consider. Defaults to None.
        rounds : int, optional
            Number of rounds for CV. Defaults to 10.
        splits : int, optional
            Number of CV splits. Defaults to 5.
        direction : str, optional
            Direction for optimization ('maximize' or 'minimize'). Defaults to 'maximize'.
        n_trials : int, optional
            Number of trials for Bayesian optimization or Random search. Defaults to 100.
        estimator_name : str, optional
            Name of the estimator to tune. Defaults to None.
        evaluation : str, optional
            Evaluation strategy. Defaults to 'cv_rounds'.
        num_features : int, optional
            Number of features to select. Defaults to None.
        feature_selection_type : str, optional
            Method for feature selection. Defaults to 'mrmr'.
        feature_selection_method : str, optional
            Inner feature selection method. Defaults to 'chi2'.
        missing_values_method : str, optional
            Strategy for handling missing values. Defaults to 'median'.
        boxplot : bool, optional
            Generate boxplots for results. Defaults to True.
        calculate_shap : bool, optional
            Compute SHAP values. Defaults to False.
        param_grid : dict, optional
            Parameter grid for the estimator. Defaults to None.
        normalization : str, optional
            Normalization strategy ('minmax' or 'standard'). Defaults to 'minmax'.
        inner_selection : str, optional
            Inner selection strategy. Defaults to 'validation_score'.
        extra_metrics : list, optional
            Additional metrics to calculate. Defaults to a comprehensive set.
        warnings_filter : bool, optional
            Suppress warnings if True. Defaults to False.
        info_to_db : bool, optional
            Save results to the database. Defaults to False.
        class_balance : str, optional
            Class balancing method. Defaults to None.
        processors : int, optional
            Number of processors to use. Defaults to -1 (use all).

        Returns:
        --------
        tuple
            Best estimator, scores DataFrame, and SHAP values (if computed).
        """
        if search_type not in ["random_search", "grid_search", "bayesian_search"]:
            raise ValueError(
                f"Invalid search type: {search_type}. Use ['random_search', 'grid_search', 'bayesian_search']."
            )
        self.model_selection_way = search_type

        self.config_cv = locals()
        self.config_cv.pop("self", None)
        self.config_cv = _validation(
            self.config_cv, self.model_selection_way, self.X, self.csv_dir, self.label, self.available_clfs
        )

        # Preprocess data
        X, _, num_feature = _preprocess(
            self.X, self.y, self.config_cv['num_features'], self.config_cv, features_names_list=features_names_list
        )

        # Define parameter grid
        if param_grid is None:
            self.param_grid = optuna_grid["NestedCV"] if search_type == "bayesian_search" else optuna_grid["SklearnParameterGrid"]
            print("Using default parameter grid")
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid

        if warnings_filter:
            warnings.filterwarnings("ignore")

        # Set up cross-validation
        custom_cv_splits = StratifiedKFold(n_splits=splits, shuffle=True).split(X, self.y)

        processors_available = os.cpu_count()
        if processors != -1:
            if processors > processors_available:
                print(f"Warning: {processors} processors not available. Using {processors_available} instead.")
                processors = processors_available
            elif processors < 1:
                print("Warning: Processors set to < 1. Using 1 processor instead.")
                processors = 1

        # Select search strategy
        if search_type in ['random_search', 'grid_search']:
            if search_type == "random_search":
                search_cv = RandomizedSearchCV(
                    estimator=AVAILABLE_CLFS[estimator_name],
                    param_distributions=self.param_grid[estimator_name],
                    scoring=scoring,
                    cv=custom_cv_splits,
                    n_iter=n_trials,
                    n_jobs=processors,
                )
            else:
                search_cv = GridSearchCV(
                    estimator=AVAILABLE_CLFS[estimator_name],
                    param_grid=self.param_grid[estimator_name],
                    scoring=scoring,
                    cv=custom_cv_splits,
                    n_jobs=processors,
                )
            search_cv.fit(X, self.y)
            best_params = search_cv.best_params_
        else:
            from optuna.logging import set_verbosity, ERROR
            # Silence Optuna's logging
            set_verbosity(ERROR)

            search_cv = optuna.integration.OptunaSearchCV(
                estimator=self.available_clfs[estimator_name],
                scoring=scoring,
                param_distributions=self.param_grid[estimator_name],
                cv=custom_cv_splits,
                return_train_score=True,
                n_jobs=processors,
                verbose=0,
                n_trials=n_trials,
                study=optuna.create_study(direction=direction, sampler=TPESampler()),
                subsample=0.7 * X.shape[0] * (splits - 1) / splits,
            )
            search_cv.fit(X, self.y)
            model_trials = search_cv.trials_

            if inner_selection in ["one_sem", "one_sem_grd"]:
                simple_model_params = _one_sem_model(model_trials, estimator_name, X.shape[0], splits, inner_selection)
            elif inner_selection in ["gso_1", "gso_2"]:
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

        return self.best_estimator, scores_df, self.shap_values if calculate_shap else None

