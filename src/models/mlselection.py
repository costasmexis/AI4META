# Standard library imports
import os
import multiprocessing
from itertools import chain

# Numerical computing
import numpy as np
import pandas as pd
from scipy.stats import sem

# Optimization and multiprocessing
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

# Custom modules
from src.models.mlestimator import MachineLearningEstimator
from src.utils.statistics.metrics_stats import _calc_metrics_stats
from src.utils.validation.validation import _validation
from src.utils.model_selection.default_cv import _cv_loop
from src.utils.model_selection.nested_cv import _outer_loop 
from src.utils.model_selection.output_config import _name_outputs, _return_csv
from src.db.input import insert_to_db
from src.utils.plots.plots import _plot_per_clf, _histogram

class MLPipelines(MachineLearningEstimator):
    def __init__(self, label, csv_dir, database_name=None, estimator=None, param_grid=None):
        if database_name is None:
            database_name = "ai4meta.db"
        super().__init__(label, csv_dir, database_name, estimator, param_grid)   
# TODO: add the rest of the parameters
    def rcv_accel(
        self,
        rounds=10,
        exclude=None,
        search_on=None,
        num_features=None,
        feature_selection_type="mrmr",
        return_csv=True,
        feature_selection_method="chi2",
        plot="box",
        scoring="matthews_corrcoef",
        splits=5,
        freq_feat=None,
        normalization="minmax",
        missing_values_method="median",
        class_balance=None,
        sfm=False,
        extra_metrics=['roc_auc', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'f1', 'average_precision', 'specificity', 'matthews_corrcoef'],
        info_to_db=False,
        filter_csv=None,
    ):
        """
        Perform accelerated cross-validation with default hyperparameters.

        This function leverages parallel processing to speed up repeated cross-validation runs.

        Parameters:
        -----------
        rounds : int, optional
            Number of CV rounds. Defaults to 10.
        exclude : list, optional
            Classifiers to exclude. Defaults to None.
        search_on : list, optional
            Classifiers to include in search. Defaults to None.
        num_features : int, optional
            Number of features for feature selection. Defaults to None.
        feature_selection_type : str, optional
            Feature selection method. Defaults to 'mrmr'.
        return_csv : bool, optional
            Save results as CSV. Defaults to True.
        plot : str, optional
            Plot type ('box' or 'violin'). Defaults to 'box'.
        normalization : str, optional
            Normalization method ('minmax' or 'std'). Defaults to 'minmax'.
        missing_values_method : str, optional
            Missing value handling method ('mean', 'median', etc.). Defaults to 'median'.
        sfm : bool, optional
            Use SelectFromModel for feature selection. Defaults to False.
        extra_metrics : list, optional
            Additional metrics to evaluate. Defaults to a pre-defined set.
        info_to_db : bool, optional
            Save results to a database. Defaults to False.
        filter_csv : dict, optional
            Filters for the results CSV. Defaults to None.

        Returns:
        --------
        pandas.DataFrame
            Results dataframe containing CV scores and statistics.
        """
        self.config_rcv = locals()
        self.config_rcv.pop("self", None)
        self.config_rcv = _validation(self.config_rcv, 'rcv', self.X, self.csv_dir, self.label, self.available_clfs)

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
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        final_dataset_name = _name_outputs(self.config_rcv, results_dir, self.csv_dir)
        statistics_dataframe = _return_csv(final_dataset_name, scores_dataframe, self.config_rcv['extra_metrics'], filter_csv, return_csv)

        if plot:
            _plot_per_clf(scores_dataframe, plot, self.config_rcv['scoring'], final_dataset_name)

        if freq_feat:
            _histogram(scores_dataframe, final_dataset_name, freq_feat, self.config_rcv['clfs'], self.X.shape[1])

        if info_to_db:
            insert_to_db(scores_dataframe, self.config_rcv, self.database_name)

        return statistics_dataframe

    def nested_cv(
        self,
        n_trials: int = 100,
        rounds: int = 10,
        exclude: str | list = None,
        search_on: str | list = None,
        info_to_db: bool = False,
        num_features: int | list = None,
        feature_selection_type: str = "mrmr",
        feature_selection_method: str = "chi2",
        sfm: bool = False,
        freq_feat: int = None,
        class_balance: str = None,
        inner_scoring: str = "matthews_corrcoef",
        outer_scoring: str = "matthews_corrcoef",
        inner_selection: list = ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"],
        extra_metrics: str | list = [
            'recall', 'specificity', 'accuracy', 'balanced_accuracy',
            'precision', 'f1', 'roc_auc', 'average_precision', 'matthews_corrcoef'
        ],
        plot: str = "box",
        inner_splits: int = 5,
        outer_splits: int = 5,
        parallel: str = "thread_per_round",
        normalization: str = "minmax",
        missing_values_method: str = "median",
        return_csv: bool = True,
        filter_csv: dict = None
    ):
        """
        Perform nested cross-validation with feature selection and hyperparameter tuning.

        Parameters:
        -----------
        n_trials : int, optional
            Number of optuna trials for hyperparameter optimization, by default 100.
        rounds : int, optional
            Number of outer cross-validation rounds, by default 10.
        exclude : str | list, optional
            List of classifiers to exclude, by default None.
        search_on : str | list, optional
            List of classifiers to search on, by default None.
        info_to_db : bool, optional
            Save results to the database, by default False.
        num_features : int | list, optional
            Number of features for selection, by default None (use all features).
        feature_selection_type : str, optional
            Feature selection type (e.g., 'mrmr', 'kbest'), by default 'mrmr'.
        feature_selection_method : str, optional
            Method for feature selection (e.g., 'chi2', 'f_classif'), by default 'chi2'.
        sfm : bool, optional
            Use SelectFromModel for supported classifiers, by default False.
        freq_feat : int, optional
            Number of top features to visualize, by default None.
        class_balance : str, optional
            Class balancing method (e.g., 'smote'), by default None.
        inner_scoring : str, optional
            Metric for inner CV loop, by default 'matthews_corrcoef'.
        outer_scoring : str, optional
            Metric for outer CV loop, by default 'matthews_corrcoef'.
        inner_selection : list, optional
            Methods for selecting hyperparameters, by default all available methods.
        extra_metrics : str | list, optional
            List of additional metrics to calculate, by default a comprehensive set.
        plot : str, optional
            Type of plot to generate ('box' or 'violin'), by default 'box'.
        inner_splits : int, optional
            Number of splits for inner CV, by default 5.
        outer_splits : int, optional
            Number of splits for outer CV, by default 5.
        parallel : str, optional
            Parallelization method ('thread_per_round' or 'freely_parallel'), by default 'thread_per_round'.
        normalization : str, optional
            Data normalization method ('minmax' or 'std'), by default 'minmax'.
        missing_values_method : str, optional
            Method for handling missing values ('mean', 'median', etc.), by default 'median'.
        return_csv : bool, optional
            Save results as a CSV, by default True.
        filter_csv : dict, optional
            Filters to apply to the results CSV, by default None.

        Returns:
        --------
        pandas.DataFrame
            Dataframe containing the results of nested cross-validation.
        """
        self.config_rncv = locals()
        self.config_rncv.pop("self", None)
        self.config_rncv = _validation(self.config_rncv, 'rncv', self.X, self.csv_dir, self.label, self.available_clfs)

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
        for inner_selection in inner_selection:
            df_inner = df[df["Inner_selection_mthd"] == inner_selection]
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
                        "Clf": classif,
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
                        "In_sel": inner_selection,
                        "Classif_rates": samples_classification_rates
                    }
                )

                # Update results with calculated metrics
                results = _calc_metrics_stats(self.config_rncv['extra_metrics'], results, indices)

        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)

        # Save results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        final_dataset_name = _name_outputs(self.config_rncv, results_dir, self.csv_dir)

        # Save to CSV if specified
        statistics_dataframe = _return_csv(final_dataset_name, scores_dataframe, self.config_rncv['extra_metrics'], filter_csv, return_csv)

        # Plot histogram if required
        if num_features is not None:    
            _histogram(scores_dataframe, final_dataset_name, freq_feat, self.config_rncv['clfs'], self.X.shape[1])

        # Generate performance plots
        if plot is not None:
            _plot_per_clf(scores_dataframe, plot, self.config_rncv['outer_scoring'], final_dataset_name)

        # Save results to database if specified
        if info_to_db:
            insert_to_db(scores_dataframe, self.config_rncv, database_name=self.database_name)

        return statistics_dataframe
