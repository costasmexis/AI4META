import os
import multiprocessing
from itertools import chain
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any

from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

from src.models.mlestimator import MachineLearningEstimator
from src.utils.statistics.metrics_stats import _calc_metrics_stats
from src.utils.validation.validation import ConfigValidator
from src.utils.model_selection.output_config import _name_outputs, _return_csv
from src.db.input import insert_to_db
from src.utils.plots.plots import _plot_per_clf, _histogram
from src.utils.model_selection.default_cv import _cv_loop
from src.utils.model_selection.nested_cv import _outer_loop

# Available classifiers
from src.constants.translators import AVAILABLE_CLFS


# Global logging flags
_logged_operations = {}

def _log_once(logger, operation: str, message: str) -> None:
    """Log a message only once for a specific operation."""
    if operation not in _logged_operations:
        _logged_operations[operation] = True
        logger.info(message)

class MLPipelines(MachineLearningEstimator):
    def __init__(self, label: str, csv_dir: str, database_name: Optional[str] = None, 
                 estimator: Optional[object] = None, param_grid: Optional[Dict] = None):
        """Initialize MLPipelines instance."""
        if database_name is None:
            database_name = "ai4meta.db"
        super().__init__(label, csv_dir, database_name, estimator, param_grid)   

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def rcv_accel(
        self,
        rounds: int = 10,
        exclude: Optional[List[str]] = None,
        search_on: Optional[List[str]] = None,
        num_features: Optional[int] = None,
        feature_selection_type: str = "mrmr",
        return_csv: bool = True,
        feature_selection_method: str = "chi2",
        plot: Optional[str] = "box",
        scoring: str = "matthews_corrcoef",
        splits: int = 5,
        freq_feat: Optional[int] = None,
        normalization: str = "minmax",
        missing_values_method: str = "median",
        class_balance: Optional[str] = None,
        sfm: bool = False,
        extra_metrics: List[str] = ['roc_auc', 'accuracy', 'balanced_accuracy', 'recall', 
                                   'precision', 'f1', 'average_precision', 'specificity', 
                                   'matthews_corrcoef'],
        info_to_db: bool = False,
        filter_csv: Optional[Dict] = None,
    ):
        """Perform accelerated cross-validation with default hyperparameters."""
        self.config_rcv = locals()
        self.config_rcv.pop("self", None)
        
        # Initialize the validator
        validator = ConfigValidator(available_clfs=AVAILABLE_CLFS)

        self.config_rcv = validator.validate_config(
            config=self.config_rcv,
            main_type="rcv",
            X=self.X,
            csv_dir=self.csv_dir,
            label=self.label
        )   

        # Setup parallelization
        num_cores = multiprocessing.cpu_count()
        use_cores = min(num_cores, rounds)
        avail_thr = max(1, num_cores // rounds)

        with threadpool_limits():
            list_dfs = Parallel(n_jobs=use_cores)(
                delayed(_cv_loop)(self.X, self.y, self.config_rcv, i, avail_thr) for i in range(rounds)
            )

        list_dfs_flat = list(chain.from_iterable(list_dfs))

        # Aggregate results
        df = pd.concat([pd.DataFrame(item) for item in list_dfs_flat], axis=0)

        results = []
        for classif in np.unique(df["Classifiers"]):
            indices = df[df["Classifiers"] == classif]
            filtered_scores = indices[f"{self.config_rcv['scoring']}"]
            
            # Aggregate sample classification rates
            samples_classification_rates = np.zeros(len(self.y))
            for test_part in indices["Samples_counts"]:
                samples_classification_rates = np.add(samples_classification_rates, test_part)
            samples_classification_rates /= rounds

            metrics_summary = {
                "Est": indices["Estimator"].unique()[0],
                "Clf": classif,
                "Hyp": 'Default',
                "Sel_way": indices["Way_of_Selection"].unique()[0],
                "Sel_feat": indices["Selected_Features"].values if num_features else None,
                "Fs_num": indices["Number_of_Features"].unique()[0],
                "Norm": normalization,
                "Miss_vals": missing_values_method,
                "Scoring": scoring,
                "Splits": splits,
                "Rnds": rounds,
                "In_sel": 'Validation_score',
                "Classif_rates": samples_classification_rates
            }

            metrics_summary = _calc_metrics_stats(self.config_rcv['extra_metrics'], [metrics_summary], indices)[-1]
            results.append(metrics_summary)

        scores_dataframe = pd.DataFrame(results)

        # Save results
        results_csv_dir = "results/csv"
        os.makedirs(results_csv_dir, exist_ok=True)
        final_dataset_name = _name_outputs(self.config_rcv, results_csv_dir, self.csv_dir)
        statistics_dataframe = _return_csv(final_dataset_name, scores_dataframe, self.config_rcv['extra_metrics'], filter_csv, return_csv)
        
        # Set the images directory
        results_image_dir = "results/images"
        os.makedirs(results_image_dir, exist_ok=True)
        final_image_name = _name_outputs(self.config_rcv, results_image_dir, self.csv_dir)

        if plot:
            _plot_per_clf(scores_dataframe, plot, self.config_rcv['scoring'], final_image_name)

        if freq_feat:
            _histogram(scores_dataframe, final_image_name, freq_feat, self.config_rcv['clfs'], self.X.shape[1])

        if info_to_db:
            insert_to_db(scores_dataframe, self.config_rcv, self.database_name)

        return statistics_dataframe

    def nested_cv(
        self,
        rounds: int = 10,
        exclude: Optional[List[str]] = None,
        search_on: Optional[List[str]] = None,
        num_features: Optional[List[int]] = None,
        feature_selection_type: str = "mrmr",
        return_csv: bool = True,
        feature_selection_method: str = "chi2",
        plot: Optional[str] = "box",
        scoring: str = "matthews_corrcoef",
        n_trials: int = 100,
        freq_feat: Optional[int] = None,
        normalization: str = "minmax",
        missing_values_method: str = "median",
        class_balance: Optional[str] = None,
        inner_scoring: str = "matthews_corrcoef",
        outer_scoring: str = "matthews_corrcoef",
        inner_selection: List[str] = ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"],
        inner_splits: int = 5,
        outer_splits: int = 5,
        parallel: str = "thread_per_round",

        sfm: bool = False,
        extra_metrics: List[str] = ['roc_auc', 'accuracy', 'balanced_accuracy', 'recall', 
                                   'precision', 'f1', 'average_precision', 'specificity', 
                                   'matthews_corrcoef'],
        info_to_db: bool = False,
        filter_csv: Optional[Dict] = None,
    ):
        """Perform nested cross-validation with feature selection and hyperparameter tuning."""
        self.config_rncv = locals()
        self.config_rncv.pop("self", None)
        
        # Initialize the validator
        validator = ConfigValidator(available_clfs=AVAILABLE_CLFS)

        self.config_rncv = validator.validate_config(
            config=self.config_rncv,
            main_type="rncv",
            X=self.X,
            csv_dir=self.csv_dir,
            label=self.label
        )   

        # Set up parallelization
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        use_cores = min(num_cores, rounds)
        
        if self.config_rncv['parallel'] == "thread_per_round":
            avail_thr = 1
            with threadpool_limits(limits=avail_thr):
                list_dfs = Parallel(n_jobs=use_cores)(
                    delayed(_outer_loop)(self.X, self.y, self.config_rncv, i, avail_thr) for i in trial_indices
                )
        else:         
            avail_thr = max(1, num_cores // rounds)
            with threadpool_limits():
                list_dfs = Parallel(n_jobs=use_cores)(
                    delayed(_outer_loop)(self.X, self.y, self.config_rncv, i, avail_thr) for i in trial_indices
                )

        # Flatten results from parallel processing
        list_dfs_flat = list(chain.from_iterable(list_dfs))

        # Create results DataFrame
        df = pd.concat([pd.DataFrame(item) for item in list_dfs_flat], axis=0)

        results = []
        for inner in inner_selection:
            df_inner = df[df["Inner_selection_mthd"] == inner]
            for classif in np.unique(df_inner["Classifiers"]):
                indices = df_inner[df_inner["Classifiers"] == classif]
                filtered_scores = indices[f"{self.config_rncv['outer_scoring']}"]
                filtered_features = indices["Selected_Features"].values if num_features is not None else None

                # Extract other details
                Numbers_of_Features = indices["Number_of_Features"].unique()[0]
                Way_of_Selection = indices["Way_of_Selection"].unique()[0]

                # Aggregate sample classification rates
                samples_classification_rates = np.zeros(len(self.y))
                for test_part in indices["Samples_counts"]:
                    samples_classification_rates = np.add(samples_classification_rates, test_part)
                samples_classification_rates /= rounds

                # Append the results
                results.append(
                    {
                        "Est": df_inner[df_inner["Classifiers"] == classif]["Estimator"].unique()[0],
                        "Clf": classif+"_"+inner,
                        "Hyp": df_inner[df_inner["Classifiers"] == classif]["Hyperparameters"].values,
                        "Sel_way": Way_of_Selection,
                        "Fs_inner": 'none' if Way_of_Selection == 'none' else feature_selection_method,
                        "Fs_num": Numbers_of_Features,
                        "Sel_feat": filtered_features,
                        "Norm": normalization,
                        "Miss_vals": missing_values_method,
                        "In_cv": inner_splits,
                        "Out_cv": outer_splits,
                        "Rnds": rounds,
                        "Trials": n_trials,
                        "Class_blnc": self.config_rncv['class_balance'],
                        "In_scor": inner_scoring,
                        "Out_scor": outer_scoring,
                        "In_sel": inner,
                        "Classif_rates": samples_classification_rates
                    }
                )

                # Update results with calculated metrics
                results = _calc_metrics_stats(self.config_rncv['extra_metrics'], results, indices)

        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)

        # Save results directory
        results_csv_dir = "results/csv"
        os.makedirs(results_csv_dir, exist_ok=True)
        final_csv_dataset_name = _name_outputs(self.config_rncv, results_csv_dir, self.csv_dir)

        results_img_dir = "results/images"
        os.makedirs(results_img_dir, exist_ok=True)
        final_img_dataset_name = _name_outputs(self.config_rncv, results_img_dir, self.csv_dir)

        # Save to CSV if specified
        statistics_dataframe = _return_csv(final_csv_dataset_name, scores_dataframe, self.config_rncv['extra_metrics'], filter_csv, return_csv)

        # Plot histogram if required
        if num_features is not None:    
            _histogram(scores_dataframe, final_img_dataset_name, freq_feat, self.config_rncv['clfs'], self.X.shape[1])

        # Generate performance plots
        if plot is not None:
            _plot_per_clf(scores_dataframe, plot, self.config_rncv['outer_scoring'], final_img_dataset_name)

        # Save results to database if specified
        if info_to_db:
            insert_to_db(scores_dataframe, self.config_rncv, database_name=self.database_name)

        return statistics_dataframe
