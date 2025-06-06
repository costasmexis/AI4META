import logging
import os
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import warnings
from typing import Optional, Dict, Any, List, Union, Tuple, Literal
import pickle
import json
import dataclasses

from src.constants.parameters_grid import optuna_grid
from src.constants.translators import AVAILABLE_CLFS, INNER_SELECTION_METHODS
from src.data.process import DataProcessor
from src.utils.statistics.metrics_stats import _calc_metrics_stats
from src.utils.model_selection.output_config import _return_csv
from src.utils.validation.dataclasses import ModelEvaluationConfig, ModelTuning
from src.database.manager import DatabaseManager

# from src.utils.validation.validation import ConfigValidator
from src.utils.model_manipulation.model_instances import _create_model_instance
from src.utils.model_manipulation.inner_selection import _one_sem_model, _gso_model
from src.utils.model_evaluation.evaluation import _evaluate
from src.utils.plots.plots import _plot_per_metric

from optuna.logging import set_verbosity, ERROR
from tqdm import tqdm
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

    def tune_cv(
        self,
        estimator_name: str,
        search_type: str = "bayesian",
        scoring: str = "matthews_corrcoef",
        features_name_list: Optional[List[str]] = None,
        splits: int = 5,
        n_trials: int = 100,
        param_grid: Optional[Dict] = None,
        inner_selection: str = "validation_score",
        info_to_db: bool = False,
        processors: int = -1,
        save_fitted_model: bool = True
    ) -> Dict[Literal['model_path', 'params_path', 'fitted_model_path', 'metadata_path'], str]:
        """Perform hyperparameter optimization using specified search methods.

        Returns:
            Dict[str, str]: Dictionary with keys: 'model_path', 'params_path', 
                        'fitted_model_path', 'metadata_path'
        """

        # Create and validate the configuration
        config = ModelTuning(
            search_type=search_type,
            scoring=scoring,
            normalization=self.normalization,
            missing_values_method=self.mv_method,
            class_balance=self.class_balance_method,
            features_name_list=features_name_list,
            splits=splits,
            n_trials=n_trials,
            estimator_name=estimator_name,
            param_grid=param_grid,
            inner_selection=inner_selection,
            info_to_db=info_to_db,
            processors=processors,
            save_fitted_model=save_fitted_model
        ).validate(self.X, self.csv_dir)

        # Preprocess data
        X, y, _ = self.process_data(
            self.X,
            self.y,
            num_features=self.X.shape[1],
            features_name_list=config.features_name_list,
            sfm=False,
            estimator_name=None
        )

        # Set up cross-validation
        custom_cv_splits = StratifiedKFold(n_splits=splits, shuffle=True).split(X, y)

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
            search_cv.fit(X, y)
            self.best_params = search_cv.best_params_
            logging.info("✓ Completed grid/random search")

        else:          
            # Suppress Optuna's verbose logging
            set_verbosity(optuna.logging.INFO)
            set_verbosity(ERROR)

            # Create a progress bar with custom format
            pbar = tqdm(total=config.n_trials, desc="Hyperparameter Search")

            def update_progress(study, trial):
                pbar.update(1)
                
                # Prepare display info
                current_value = f'{trial.value:.4f}' if trial.value is not None else 'Failed'
                best_value = f'{study.best_value:.4f}' if study.best_value is not None else 'N/A'
                best_trial_num = study.best_trial.number if study.best_trial is not None else 'N/A'
                
                # Update postfix with detailed info
                pbar.set_postfix({
                    'Trial': trial.number,
                    'Current': current_value,
                    'Best': best_value,
                    'Best Trial': best_trial_num
                })

            search_cv = optuna.integration.OptunaSearchCV(
                estimator=config.estimator,
                scoring=config.scoring,
                param_distributions=config.param_grid,
                cv=custom_cv_splits,
                return_train_score=True,
                n_jobs=config.processors,
                verbose=0,
                n_trials=config.n_trials,
                study=optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler()),
                subsample=0.7 * X.shape[0] * (splits - 1) / splits,
                callbacks=[update_progress]
            )

            try:
                search_cv.fit(X, y)
            finally:
                pbar.close()
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

        # Save the final model
        with open(f"{config.model_path}", "wb") as model_file:
            pickle.dump(self.best_model, model_file)
        self.logger.info(f"✓ Model saved to {config.model_path}")
        
        # Save a json with the best hyperparameters and the estimator
        with open(f"{config.params_path}", "w") as params_file:
            json.dump(self.best_params, params_file)
        self.logger.info(f"✓ Best hyperparameters saved to {config.params_path}")

        # If save_fitted_model is True, fit the model on the entire dataset
        if config.save_fitted_model:
            fit_model = self.best_model.fit(X, y)
            with open(f"{config.fitted_model_path}", "wb") as fit_model_file:
                pickle.dump(fit_model, fit_model_file)
            self.logger.info(f"✓ Fitted model path: {config.fitted_model_path}")
        
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

        return {
            'model_path': config.model_path,
            'params_path': config.params_path,
            'fitted_model_path': config.fitted_model_path,
            'metadata_path': config.metadata_path
        }

    def evaluation(
        self,
        model_path: str = None,
        evaluation: str = 'cv_rounds',
        prefitted: bool = False,
        rounds: int = None,
        splits: int = None,
        info_to_db: bool = False,
        calculate_shap: bool = False,
        boxplot: bool = True,
        features_name_list: Optional[List[str]] = None
    ) -> Dict[Literal['eval_csv_path'], str]:
        """Evaluate the model using specified evaluation methods.
        Returns:
            Dict[str, str]: Dictionary with key: 'eval_csv_path'
        """

        # Create and validate the configuration
        config = ModelEvaluationConfig(
            estimator_name=None,
            inner_selection=None,
            model_path=model_path,
            evaluation=evaluation,
            calculate_shap=calculate_shap,
            rounds=rounds,
            splits=splits,
            normalization=self.normalization,
            missing_values_method=self.mv_method,
            extra_metrics=self.extra_metrics,
            class_balance=self.class_balance_method,
            features_name_list=features_name_list,
            info_to_db=info_to_db,
            boxplot=boxplot
        ).validate(self.X, self.csv_dir)

        # Preprocess data
        X, y, _ = self.process_data(
            self.X,
            self.y,
            num_features=self.X.shape[1],
            features_name_list=config.features_name_list,
            sfm=False,
            estimator_name=None
        )

        # Import the best model from the model path from the .pkl file  
        with open(config.model_path, "rb") as model_file:
            self.best_model = pickle.load(model_file)

        # Assign the estimator name to the config
        config.estimator_name = type(self.best_model).__name__
            
        # Apply evaluation in the final model
        scores_df, shaps_array = _evaluate(X, y, self.best_model, config, self)

        # Generate visualizations and save results
        if config.boxplot:
            _plot_per_metric(scores_df, config.dataset_plot_name)
        
        # Save results to CSV
        results = [
            {
                "Est": config.estimator_name,
                "Norm": config.normalization,
                "Miss_vals": config.missing_values_method,
                "Eval": config.evaluation,
                "Class_blnc": config.class_balance,
            }
        ]
        stat_lst = _calc_metrics_stats(config.extra_metrics, results=results, indices=scores_df)
        # Save the results to a CSV file
        stat_df = pd.DataFrame(stat_lst)
        stat_df = stat_df.drop(columns=config.extra_metrics, axis=1)
        stat_df.to_csv(f"{config.dataset_csv_name}")
        self.logger.info(f"✓ Results saved to {config.dataset_csv_name}")      

        # Save boxplot 

        if calculate_shap:
            self.shap_values = shaps_array
            self.logger.info("✓ SHAP values calculated")
        
                # Save results to database if specified
        if info_to_db:
            dbman = DatabaseManager(self.database_name)
            # Create a new single-row dataframe to hold consolidated results
            consolidated_df = pd.DataFrame()
            # Format hyperparameters as an array containing a single dictionary
            # Add best model parameters as a dictionary
            if hasattr(self.best_model, "get_params"):
                consolidated_df["Hyp"] = [self.best_model.get_params()]
            else:
                logging.warning("Best model does not have get_params method. Using best_params from the last tune_cv application instead.")
                consolidated_df["Hyp"] = [self.best_params]
            # Add consolidated metrics - store each metric as a list of values
            for column in scores_df.columns:
                # Store the list of values for each metric
                consolidated_df[column] = [scores_df[column].tolist()]
            # Add the shap values if calculated
            if calculate_shap:
                consolidated_df["Shap"] = [self.shap_values]
            dbman.insert_experiment_data(consolidated_df, config, database_name=self.database_name)

        logging.info("✓ Model evaluation completed")

        return {
            'eval_csv_path': config.dataset_csv_name
        }