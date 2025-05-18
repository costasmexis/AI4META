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
import dataclasses

from src.constants.parameters_grid import optuna_grid
from src.constants.translators import AVAILABLE_CLFS, SFM_COMPATIBLE_ESTIMATORS
from src.data.process import DataProcessor
from src.utils.statistics.metrics_stats import _calc_metrics_stats
from src.utils.model_selection.output_config import _return_csv
from src.utils.validation.dataclasses import ModelEvaluationConfig
from src.database.manager import DatabaseManager

# from src.utils.validation.validation import ConfigValidator
from src.utils.model_manipulation.model_instances import _create_model_instance
from src.utils.model_manipulation.inner_selection import _one_sem_model, _gso_model
from src.utils.model_evaluation.evaluation import _evaluate
from src.utils.plots.plots import _plot_per_metric
# from src.db.input import insert_to_db
# from src.utils.statistics.metrics_stats import _calc_metrics_stats

class MachineLearningEstimator(DataProcessor):
    """Class for machine learning model estimation and evaluation."""

    def __init__(
            self,
            label: str,
            csv_dir: str,
            index_col: Optional[str] = None,
            normalization: str = "minmax",
            mv_method: str = "median",
            fs_method: str = "mrmr",
            inner_fs_method: str = "chi2",
            class_balance_method: Optional[str] = None,
            database_name: Optional[str] = None,
            preprocess_mode: str = "general",
            extra_metrics: Optional[List[str]] = [
                'roc_auc', 'recall', 'matthews_corrcoef', 'accuracy', 'balanced_accuracy',
                'precision', 'f1', 'average_precision', 'specificity'
            ]        
        )-> None:
        """
        Initialize the MachineLearningEstimator instance.
        """
        super().__init__(
            label=label,
            csv_dir=csv_dir,
            index_col=index_col,
            normalization=normalization,
            mv_method=mv_method,
            fs_method=fs_method,
            inner_fs_method=inner_fs_method,
            class_balance_method=class_balance_method,
            preprocess_mode=preprocess_mode
        )

        self.database_name = database_name
        self.extra_metrics = extra_metrics
        self.best_params = None
        self.best_model = None
        self.shap_values = None

        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for the class."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("MachineLearningEstimator initialized with parameters:")
        self.logger.info(f"Label: {self.label}")
        self.logger.info(f"CSV Directory: {self.csv_dir}")
        self.logger.info(f"Index Column: {self.index_col}")
        self.logger.info(f"Normalization: {self.normalization}")
        self.logger.info(f"Feature Selection Method: {self.fs_method}")
        self.logger.info(f"Missing Values Method: {self.mv_method}")
        self.logger.info(f"Database Name: {self.database_name}")

    def search_cv(
        self,
        estimator_name: str,
        search_type: str = "bayesian",
        scoring: str = "matthews_corrcoef",
        features_name_list: Optional[List[str]] = None,
        rounds: int = 20,
        splits: int = 5,
        direction: str = "maximize",
        n_trials: int = 100,
        evaluation: str = "cv_rounds",
        num_features: Optional[int] = None,
        sfm: Optional[bool] = False,
        boxplot: bool = True,
        calculate_shap: bool = False,
        param_grid: Optional[Dict] = None,
        inner_selection: str = "validation_score",
        info_to_db: bool = False,
        processors: int = -1,
        save_model: bool = True
    ):
        """Perform hyperparameter optimization using specified search methods."""
        
        # Create and validate the configuration
        config = ModelEvaluationConfig(
            search_type=search_type,
            scoring=scoring,
            normalization=self.normalization,
            feature_selection_type=self.fs_method,
            feature_selection_method=self.inner_fs_method,
            missing_values_method=self.mv_method,
            extra_metrics=self.extra_metrics,
            class_balance=self.class_balance_method,
            features_name_list=features_name_list,
            rounds=rounds,
            splits=splits,
            direction=direction,
            n_trials=n_trials,
            estimator_name=estimator_name,
            evaluation=evaluation,
            num_features=num_features,
            boxplot=boxplot,
            calculate_shap=calculate_shap,
            param_grid=param_grid,
            inner_selection=inner_selection,
            info_to_db=info_to_db,
            processors=processors,
            sfm=sfm,
            save_model=save_model
        ).validate(self.X, self.csv_dir)

        # Preprocess data
        X, _, num_feature = self.process_data(
            self.X,
            self.y,
            num_features=config.num_features,
            features_name_list=config.features_name_list,
            sfm=config.sfm,
            estimator_name=config.estimator_name
        )

        # Set up cross-validation
        custom_cv_splits = StratifiedKFold(n_splits=splits, shuffle=True).split(X, self.y)

        # Perform model selection       
        if search_type in ['random', 'grid']:
            if search_type == "random":
                search_cv = RandomizedSearchCV(
                    estimator=config.estimator,
                    param_distributions=config.param_grid,
                    scoring=config.scoring,
                    cv=custom_cv_splits,
                    n_iter=config.n_trials,
                    n_jobs=config.processors,
                )
            else:
                search_cv = GridSearchCV(
                    estimator=config.estimator,
                    param_grid=config.param_grid,
                    scoring=config.scoring,
                    cv=custom_cv_splits,
                    n_jobs=config.processors,
                )
            search_cv.fit(X, self.y)
            self.best_params = search_cv.best_params_
            logging.info("✓ Completed grid/random search")

        else:          
            from optuna.logging import set_verbosity, ERROR
            set_verbosity(ERROR)
            
            search_cv = optuna.integration.OptunaSearchCV(
                estimator=config.estimator,
                scoring=config.scoring,
                param_distributions=config.param_grid,
                cv=custom_cv_splits,
                return_train_score=True,
                n_jobs=config.processors,
                verbose=0,
                n_trials=config.n_trials,
                study=optuna.create_study(direction=config.direction, sampler=optuna.samplers.TPESampler()),
                subsample=0.7 * X.shape[0] * (splits - 1) / splits,
            )
            search_cv.fit(X, self.y)
            model_trials = search_cv.trials_

            # Select best parameters based on inner selection method            
            if inner_selection in ["one_sem", "one_sem_grd"]:
                self.best_params = _one_sem_model(model_trials, config.estimator_name, X.shape[0], config.splits, config.inner_selection)
            elif inner_selection in ["gso_1", "gso_2"]:
                self.best_params = _gso_model(model_trials, config.estimator_name, X.shape[0], config.splits, config.inner_selection)
            else:
                self.best_params = search_cv.best_params_
            logging.info("✓ Completed Optuna search")

        # Create and evaluate final model
        self.best_model = _create_model_instance(config.estimator_name, self.best_params)
        
        # Apply evaluatuin in the final model
        scores_df, shaps_array = _evaluate(X, self.y, self.best_model, self.best_params, config, self)

        # Generate visualizations and save results
        if boxplot:
            _plot_per_metric(scores_df, config.dataset_plot_name)
            self.logger.info("✓ Plots saved")
        
        # Save results to CSV
        results = [
            {
                "Est": config.estimator_name,
                "Sel_way": 'none' if config.num_features == self.X.shape[1] else config.feature_selection_type,
                "Fs_num": config.num_features,
                "Norm": config.normalization,
                "Miss_vals": config.missing_values_method,
                "Eval": config.evaluation,
                "Class_blnc": config.class_balance,
                "Scoring": config.scoring,
                "Fs_inner": config.feature_selection_method,
                "In_sel": config.inner_selection,
                "search_type": config.search_type
            }
        ]
        stat_lst = _calc_metrics_stats(config.extra_metrics, results=results, indices=scores_df)
        stat_df = pd.DataFrame(stat_lst)
        stat_df = stat_df.drop(columns=config.extra_metrics, axis=1)
        stat_df.to_csv(f"{config.dataset_csv_name}")
        self.logger.info(f"✓ Results saved to {config.dataset_csv_name}")
        # TODO: add a try except here
        # if info_to_db:
        #     scores_db_df = pd.DataFrame({col: [scores_df[col].tolist()] for col in scores_df.columns})
        #     insert_to_db(scores_db_df, self.config_cv, self.database_name)

        if save_model:
            # Save the final model
            with open(f"{config.model_path}", "wb") as model_file:
                pickle.dump(self.best_model, model_file)
            self.logger.info(f"✓ Model saved to {config.model_path}")
            
            # Save a json with the best hyperparameters and the estimator
            with open(f"{config.params_path}", "w") as params_file:
                json.dump(self.best_params, params_file)
            self.logger.info(f"✓ Best hyperparameters saved to {config.params_path}")
            
            # Save a json with the metadata of the function 
            with open(f"{config.metadata_path}", "w") as metadata_file:
                
                # After creating config_dict
                config_dict = dataclasses.asdict(config)

                # Convert param_grid to a string representation
                if 'param_grid' in config_dict and config_dict['param_grid'] is not None:
                    config_dict['param_grid'] = str(config_dict['param_grid'])

                # Also convert other non-serializable objects
                if '_logger' in config_dict:
                    del config_dict['_logger']
                if '_logged_messages' in config_dict:
                    del config_dict['_logged_messages']

                # Add the csv_dir to the config_dict
                config_dict['csv_dir'] = self.csv_dir
                config_dict['index_col'] = self.index_col

                # Now dump to JSON
                json.dump(config_dict, metadata_file, indent=2)

            self.logger.info(f"✓ Metadata of the function saved to {config.metadata_path}")

        if calculate_shap:
            self.shap_values = shaps_array
            self.logger.info("✓ SHAP values calculated")
        
                # Save results to database if specified
        if info_to_db:
            dbman = DatabaseManager(self.database_name)
            # Create a new single-row dataframe to hold consolidated results
            consolidated_df = pd.DataFrame()
            # Format hyperparameters as an array containing a single dictionary
            consolidated_df["Hyp"] = np.array([self.best_params])
            # Add feature selection information (typically 'none' for MLEstimator)
            # Check if sfm applied
            if config.sfm and config.estimator_name in SFM_COMPATIBLE_ESTIMATORS:
                consolidated_df["Sel_way"] = 'none' if config.num_features == self.X.shape[1] else 'sfm'
                consolidated_df["Fs_inner"] = 'none' if config.num_features == self.X.shape[1] else 'none'
            else:
                consolidated_df["Sel_way"] = 'none' if config.num_features == self.X.shape[1] else config.feature_selection_type
                consolidated_df["Fs_inner"] = 'none' if config.num_features == self.X.shape[1] else config.feature_selection_method
            consolidated_df["Fs_num"] = config.num_features
            # Add consolidated metrics - store each metric as a list of values
            for column in scores_df.columns:
                # Store the list of values for each metric
                consolidated_df[column] = [scores_df[column].tolist()]
            # Add the shap values if calculated
            if calculate_shap:
                consolidated_df["Shap"] = [self.shap_values]
            dbman.insert_experiment_data(consolidated_df, config, database_name=self.database_name)

        logging.info("✓ Model evaluation completed")