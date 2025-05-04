import logging
import os
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import warnings
from typing import Optional, Dict, Any, List, Union, Tuple
import pickle
import json

from src.data.dataloader import DataLoader
from src.features.features_selection import preprocess
from src.data.class_balance import _class_balance
from src.constants.parameters_grid import optuna_grid
from src.constants.translators import AVAILABLE_CLFS
from src.utils.validation.validation import ConfigValidator
from src.utils.model_manipulation.model_instances import _create_model_instance
from src.utils.model_manipulation.inner_selection import _one_sem_model, _gso_model
from src.utils.model_selection.output_config import _name_outputs
from src.utils.model_evaluation.evaluation import _evaluate
from src.utils.plots.plots import _plot_per_metric
from src.db.input import insert_to_db
from src.utils.statistics.metrics_stats import _calc_metrics_stats

# Global logging flags
_logged_operations = {}

def _log_once(logger, operation: str, message: str) -> None:
    """Log a message only once for a specific operation."""
    if operation not in _logged_operations:
        _logged_operations[operation] = True
        logger.info(message)

class MachineLearningEstimator(DataLoader):
    """Class for machine learning model estimation and evaluation."""

    def __init__(self, label: str, csv_dir: str, database_name: Optional[str] = None, 
                 estimator: Optional[object] = None, param_grid: Optional[Dict] = None):
        """Initialize the MLEstimator instance."""
        super().__init__(label, csv_dir)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.database_name = database_name or "ai4meta.db"
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
        search_type: str = "bayesian_search",
        scoring: str = "matthews_corrcoef",
        features_names_list: Optional[List[str]] = None,
        rounds: int = 20,
        splits: int = 5,
        direction: str = "maximize",
        n_trials: int = 100,
        estimator_name: Optional[str] = None,
        evaluation: str = "cv_rounds",
        num_features: Optional[int] = None,
        feature_selection_type: str = "mrmr",
        feature_selection_method: str = "chi2",
        missing_values_method: str = "median",
        boxplot: bool = True,
        calculate_shap: bool = False,
        param_grid: Optional[Dict] = None,
        normalization: str = "minmax",
        inner_selection: str = "validation_score",
        extra_metrics: List[str] = [
            'roc_auc', 'accuracy', 'balanced_accuracy', 'recall', 'precision',
            'f1', 'average_precision', 'specificity', 'matthews_corrcoef'
        ],
        warnings_filter: bool = False,
        info_to_db: bool = False,
        class_balance: Optional[str] = None,
        info_to_results: bool = True,
        processors: int = -1,
        save_model: bool = True
    ):
        """Perform hyperparameter optimization using specified search methods."""
        
        if search_type not in ["random_search", "grid_search", "bayesian_search"]:
            raise ValueError(
                f"Invalid search type: {search_type}. Use ['random_search', 'grid_search', 'bayesian_search']"
            )
        self.model_selection_way = search_type

        self.config_cv = locals()
        self.config_cv.pop("self", None)

        # Initialize the validator
        validator = ConfigValidator(available_clfs=AVAILABLE_CLFS)

        self.config_cv = validator.validate_config(
            config=self.config_cv,
            main_type=search_type,
            X=self.X,
            csv_dir=self.csv_dir,
            label=self.label
        )   

        # Preprocess data
        X, _, num_feature = preprocess(
            self.config_cv, num_features, self.X, self.y, features_names_list=features_names_list
        )

        # Configure parameter grid
        if param_grid is None:
            self.param_grid = optuna_grid["NestedCV"] if search_type == "bayesian_search" else optuna_grid["SklearnParameterGrid"]
            _log_once(self.logger, 'default_grid',
                     "✓ Using default parameter grid")
        else:
            param_grid = {estimator_name: param_grid}
            self.param_grid = param_grid

        if warnings_filter:
            warnings.filterwarnings("ignore")

        # Set up cross-validation
        custom_cv_splits = StratifiedKFold(n_splits=splits, shuffle=True).split(X, self.y)

        # Configure processors
        processors_available = os.cpu_count()
        if processors != -1:
            if processors > processors_available:
                _log_once(self.logger, 'processor_warning',
                         f"Warning: {processors} processors not available. Using {processors_available} instead.")
                processors = processors_available
            elif processors < 1:
                _log_once(self.logger, 'processor_min',
                         "Warning: Processors set to < 1. Using 1 processor instead.")
                processors = 1

        # Perform model selection       
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
            _log_once(self.logger, f'{search_type}_complete',
                     f"✓ Completed {search_type}")
        else:          
            from optuna.logging import set_verbosity, ERROR
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
                study=optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler()),
                subsample=0.7 * X.shape[0] * (splits - 1) / splits,
            )
            search_cv.fit(X, self.y)
            model_trials = search_cv.trials_

            # Select best parameters based on inner selection method            
            if inner_selection in ["one_sem", "one_sem_grd"]:
                best_params = _one_sem_model(model_trials, estimator_name, X.shape[0], splits, inner_selection)
            elif inner_selection in ["gso_1", "gso_2"]:
                best_params = _gso_model(model_trials, self.name, splits, inner_selection)
            else:
                best_params = search_cv.best_params_

            _log_once(self.logger, 'bayesian_complete',
                     "✓ Completed Bayesian optimization")

        # Create and evaluate final model
        self.config_cv['hyperparameters'] = best_params
        best_model = _create_model_instance(estimator_name, best_params)
        self.best_estimator = best_model

        scores_df, shaps_array = _evaluate(X, self.y, best_model, best_params, self.config_cv)

        # Generate visualizations and save results
        if boxplot:
            results_image_dir = "results/images"
            os.makedirs(results_image_dir, exist_ok=True)
            final_image_name = _name_outputs(self.config_cv, results_image_dir, self.csv_dir)
            _plot_per_metric(scores_df, final_image_name)
        
        if info_to_results:
            results_csv_dir = "results/csv"
            os.makedirs(results_csv_dir, exist_ok=True)

            # Initiate the results
            results = [
                {
                    "Est": estimator_name,
                    "Sel_way": 'none' if num_features is None or num_features == self.X.shape[1] else feature_selection_type,
                    "Fs_num": num_features,
                    "Norm": normalization,
                    "Miss_vals": missing_values_method,
                    "Eval": evaluation,
                    "Class_blnc": class_balance,
                    "Scoring": scoring,
                    "Fs_inner": feature_selection_method,
                    "In_sel": inner_selection,
                    "search_type": search_type
                }
            ]
            # stat_df = final_model_stats(results_df, evaluation=evaluation)
            stat_lst = _calc_metrics_stats(extra_metrics, results=results, indices=scores_df)
            stat_df = pd.DataFrame(stat_lst)
            stat_df = stat_df.drop(columns=extra_metrics, axis=1)
            results_name = _name_outputs(self.config_cv, results_csv_dir, self.csv_dir)
            stat_df.to_csv(f"{results_name}_final_model.csv")
            self.logger.info(f"✓ Results saved to {results_name}_final_model.csv")
        # TODO: add a try except here
        if info_to_db:
            scores_db_df = pd.DataFrame({col: [scores_df[col].tolist()] for col in scores_df.columns})
            insert_to_db(scores_db_df, self.config_cv, self.database_name)

        if save_model:
            model_dir = "results/models"
            os.makedirs(model_dir, exist_ok=True)
            model_name = _name_outputs(self.config_cv, model_dir, self.csv_dir)

            # Save the final model
            with open(f"{model_name}_final_model.pkl", "wb") as model_file:
                pickle.dump(best_model, model_file)
            self.logger.info(f"✓ Model saved to {model_name}_final_model.pkl")
            
            # Save a json with the best hyperparameters and the estimator
            with open(f"{model_name}_best_params.json", "w") as params_file:
                json.dump(best_params, params_file)
            self.logger.info(f"✓ Best hyperparameters saved to {model_name}_best_params.json")
            
            # Save a json with the metadata of the function 
            with open(f"{model_name}_metadata.json", "w") as metadata_file:
                json.dump(self.config_cv, metadata_file)
            self.logger.info(f"✓ Metadata of the function saved to {model_name}_metadata.json")

        if calculate_shap:
            self.shap_values = shaps_array
            
        _log_once(self.logger, 'complete',
                 "✓ Model selection and evaluation complete")

        return self.best_estimator, scores_df, self.shap_values if calculate_shap else None