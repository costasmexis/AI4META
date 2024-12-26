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
from src.utils.validation import _validation
from src.utils.model_selection.default_cv import _cv_loop
from src.utils.model_selection.nested_cv import _outer_loop 
from src.utils.model_selection.results_config import _name_outputs, _return_csv
from src.utils.model_selection.database_input import insert_to_db
from src.utils.plots import _plot_per_clf, _histogram

class MLPipelines(MachineLearningEstimator):
    def __init__(self, label, csv_dir, database_name=None, estimator=None, param_grid=None):
        if database_name is None:
            database_name = "ai4meta.db"
        super().__init__(label, csv_dir, database_name, estimator, param_grid)

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
        class_balance = None,
        sfm=False,
        extra_metrics=['roc_auc','accuracy','balanced_accuracy','recall','precision','f1','average_precision','specificity'],
        info_to_db=False,
        filter_csv=None
    ):
        """
        Accelerated version of the nested cross-validation process.

        This function performs nested cross-validation faster by using parallel
        processing. It is the same as the nested cross-validation process, but
        it uses the joblib library to use multiple cores of the computer.

        Parameters
        ----------
        rounds : int, optional
            Number of rounds to perform the nested cross-validation. The
            default is 10.
        exclude : list, optional
            List of estimators to exclude from the nested cross-validation.
            The default is None.
        search_on : str, optional
            Estimator to search for hyperparameters. The default is None.
        num_features : int, optional
            Number of features to select. The default is None.
        feature_selection_type : str, optional
            Method to select features. The default is 'mrmr'.
        return_csv : bool, optional
            Whether to return the results as a CSV file. The default is True.
        feature_selection_method : str, optional
            Method to select features. The default is 'chi2'.
        plot : str, optional
            Whether to plot the results. The default is 'box'.
        scoring : str, optional
            Scoring method to use. The default is 'matthews_corrcoef'.
        splits : int, optional
            Number of splits to perform in the nested cross-validation. The
            default is 5.
        normalization : str, optional
            Method of normalization, either 'minmax' or 'std' or None, by default 'minmax'
        missing_values_method : str, optional
            Method of handling missing values, either 'median' or 'mean' or '0' or 'drop' or None, by default 'median'
        num_features : int|list, optional
            Number of features to select, either a single number or a list of numbers, by default None (all features)
        feature_selection_type : str, optional
            Type of feature selection, either 'mrmr' or 'kbest' or 'percentile', by default 'mrmr' (if no number of features is provided its not in use)
        feature_selection_method : str, optional
            Method of feature selection, either 'chi2', 'mutual_info_classif', 'f_classif', by default 'chi2' (if no number of features is provided its not in use)
        sfm : bool, optional
            If True, the feature selection is done using the SelectFromModel class only for the estimators that support it. The rest of the estimators are using the feature_selection_type. By default False
        freq_feat : int, optional
            A histogram of the frequency of the (freaq_feat or all if None) features will be plotted, by default None
        class_balance : str, optional
            If not None, class balancing will be applied using 'smote', 'borderline_smote', or 'tomek'. By default None
        extra_metrics : list, optional
            List of extra metrics to calculate. The default is
            ['roc_auc','accuracy','balanced_accuracy','recall','precision',
            'f1','average_precision','specificity'].
        info_to_db : bool, optional
            Whether to add the results to a database. The default is False.
        filter_csv : list, optional
            List of columns to filter from the CSV file. The default is None.

        Returns
        -------
        results : pandas dataframe
            Dataframe with the results of the nested cross-validation.
        """
        self.config_rcv = locals()
        self.config_rcv.pop("self", None)
        self.config_rcv = _validation(self.config_rcv,'rcv', self.X, self.csv_dir, self.label, self.available_clfs)
        
        # Parallelization
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        if num_cores < rounds:
            use_cores = num_cores
        else:   
            use_cores = rounds
        avail_thr = max(1, num_cores // rounds)

        with threadpool_limits():
            list_dfs = Parallel(n_jobs=use_cores,verbose=0)(
                delayed(_cv_loop)(self.X, self.y, self.config_rcv,i,avail_thr) for i in trial_indices
            )

        list_dfs_flat = list(chain.from_iterable(list_dfs))
        
         # Create results dataframe
        results = []
        df = pd.DataFrame()

        for item in list_dfs_flat:
            dataframe = pd.DataFrame(item)
            df = pd.concat([df, dataframe], axis=0)

        for classif in np.unique(df["Classifiers"]):
            indices = df[df["Classifiers"] == classif]
            filtered_scores = indices[f"{self.config_rcv['scoring']}"].values
            if num_features is not None:
                filtered_features = indices["Selected_Features"].values
            mean_score = np.mean(filtered_scores)
            max_score = np.max(filtered_scores)
            std_score = np.std(filtered_scores)
            sem_score = sem(filtered_scores)
            median_score = np.median(filtered_scores)
            Numbers_of_Features = indices["Number_of_Features"].unique()[0]
            Way_of_Selection = indices["Way_of_Selection"].unique()[0]
            samples_classification_rates = np.zeros(len(self.y))
            for test_part in indices["Samples_counts"]:
                samples_classification_rates=np.add(samples_classification_rates,test_part)
            samples_classification_rates /= rounds
                            
            results.append(
                {
                    "Est": df[df["Classifiers"] == classif]["Estimator"].unique()[
                        0
                    ],
                    "Clf": classif,
                    "Hyp": 'Default',
                    "Sel_feat": filtered_features
                    if num_features is not None
                    else None,
                    "Fs_num": Numbers_of_Features,
                    "Sel_way": Way_of_Selection,
                    "Fs_inner": 'none' if Way_of_Selection=='none' else feature_selection_method,
                    "Norm": normalization,
                    "Miss_vals": missing_values_method,
                    "Splits": splits,
                    "Rnds": rounds,
                    "Class_blnc": self.config_rcv['class_balance'],
                    "Scoring": scoring,
                    "In_sel": 'validation_score',
                    "Classif_rates": samples_classification_rates.tolist(),
                }
            )
            
            results = _calc_metrics_stats(
                self.config_rcv['extra_metrics'], results, indices
            )
                                            
        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)

        # Create a 'Results' directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Initiate name
        final_dataset_name = _name_outputs(self.config_rcv, results_dir, self.csv_dir)  
          
        # Save the results to a CSV file of the outer scores for each classifier
        # if return_csv:
        statistics_dataframe = _return_csv(final_dataset_name, scores_dataframe, self.config_rcv['extra_metrics'], filter_csv, return_csv)

        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:
            # Plot histogram of features
            _histogram(scores_dataframe, final_dataset_name, freq_feat, self.config_rcv['clfs'], self.X.shape[1])
        
        # Plot box or violin plots of the outer cross-validation scores 
        if plot is not None:
            _plot_per_clf(scores_dataframe, plot, self.config_rcv['scoring'], final_dataset_name)
            
        if info_to_db:
            # Add to database
            insert_to_db(scores_dataframe, self.config_rcv, self.database_name)

        return statistics_dataframe

    def nested_cv(
        self,
        n_trials: int = 100,
        rounds: int = 10,
        exclude: str|list = None,
        search_on: str|list = None,
        info_to_db: bool = False,
        num_features: int|list = None,
        feature_selection_type: str = "mrmr",
        feature_selection_method: str = "chi2",
        sfm: bool = False,
        freq_feat: int = None,
        class_balance: str = None,
        inner_scoring: str = "matthews_corrcoef",
        outer_scoring: str = "matthews_corrcoef",
        inner_selection: list = ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"],
        extra_metrics: str|list = ['recall', 'specificity', 'accuracy', 'balanced_accuracy', 
                            'precision', 'f1', 'roc_auc', 'average_precision', 'matthews_corrcoef'],
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
        Perform nested cross-validation with optional feature selection and hyperparameter optimization.

        Parameters
        ----------
        n_trials : int, optional
            Number of optuna trials for hyperparameter optimization, by default 100
        rounds : int, optional
            Number of outer cross-validation rounds, by default 10
        exclude : str|list, optional
            List of classifiers (or string with one classifier) to exclude from the search, by default None
        search_on : str|list, optional
            List of classifiers (or string with one classifier) to search for hyperparameters on, by default None
        info_to_db : bool, optional
            If True, the results will be added to the database, by default False
        num_features : int|list, optional
            Number of features to select, either a single number or a list of numbers, by default None (all features)
        feature_selection_type : str, optional
            Type of feature selection, either 'mrmr' or 'kbest' or 'percentile', by default 'mrmr' (if no number of features is provided its not in use)
        feature_selection_method : str, optional
            Method of feature selection, either 'chi2', 'mutual_info_classif', 'f_classif', by default 'chi2' (if no number of features is provided its not in use)
        sfm : bool, optional
            If True, the feature selection is done using the SelectFromModel class only for the estimators that support it. The rest of the estimators are using the feature_selection_type. By default False
        freq_feat : int, optional
            A histogram of the frequency of the (freaq_feat or all if None) features will be plotted, by default None
        class_balance : str, optional
            If not None, class balancing will be applied using 'smote', 'borderline_smote', or 'tomek'. By default None
        inner_scoring : str, optional
            Scoring metric used in the inner cross-validation loop, by default 'matthews_corrcoef'
        outer_scoring : str, optional
            Scoring metric used in the outer cross-validation loop, by default 'matthews_corrcoef'
        inner_selection : list, optional
            List of methods for selecting the hyperparameters, by default ['validation_score', 'one_sem', 'gso_1', 'gso_2', 'one_sem_grd']
            Should be a subset of ['validation_score', 'one_sem', 'gso_1', 'gso_2', 'one_sem_grd'] to be valid.
        extra_metrics : str|list, optional
            List of extra metrics (or string with one metric) to calculate, by default ['recall', 'specificity', 'accuracy', 'balanced_accuracy', 
            'precision', 'f1', 'roc_auc', 'average_precision', 'matthews_corrcoef']
        plot : str, optional
            Type of plot to create, either 'box' or 'violin', by default 'box'
        inner_splits : int, optional
            Number of splits in the inner cross-validation loop, by default 5
        outer_splits : int, optional
            Number of splits in the outer cross-validation loop, by default 5
        parallel : str, optional
            Method of parallelization, either 'thread_per_round' or 'freely_parallel', by default 'thread_per_round'
            If the user needs resources for other procedures, it is recommended to use 'thread_per_round'
        normalization : str, optional
            Method of normalization, either 'minmax' or 'std' or None, by default 'minmax'
        missing_values_method : str, optional
            Method of handling missing values, either 'median' or 'mean' or '0' or 'drop' or None, by default 'median'
        return_csv : bool, optional
            If True, the results will be returned as a CSV file with the statistics of the results, by default True
        filter_csv : dict, optional
            Dictionary of filters to apply to the trials of the results dataframe and csv, by default None

        Returns
        -------
        pandas.DataFrame
            A dataframe with the results of the nested cross-validation.
        
            An updated database if info_to_db is True
            Plots if plot is not None
            Histogram of the frequency of the features if freq_feat is not None
        """
        self.config_rncv = locals()
        self.config_rncv.pop("self", None)
        self.config_rncv = _validation(self.config_rncv,'rncv', self.X, self.csv_dir, self.label, self.available_clfs)
        
        # Parallelization
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        if num_cores < rounds:
            use_cores = num_cores
        else:
            use_cores = rounds


        if self.config_rncv['parallel'] == "thread_per_round":
            avail_thr = 1
            with threadpool_limits(limits=avail_thr):
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(_outer_loop)(self.X, self.y, self.config_rncv, i, avail_thr) for i in trial_indices
                )
        else:         
            avail_thr = max(1, num_cores // rounds)
            with threadpool_limits():
                list_dfs = Parallel(n_jobs=use_cores, verbose=0)(
                    delayed(_outer_loop)(self.X, self.y, self.config_rncv, i, avail_thr) for i in trial_indices
                )

        list_dfs_flat = list(chain.from_iterable(list_dfs))
        
        # Create results dataframe
        results = []
        df = pd.DataFrame()
        for item in list_dfs_flat:
            dataframe = pd.DataFrame(item)
            df = pd.concat([df, dataframe], axis=0)

        for inner_selection in inner_selection:
            df_inner = df[df["Inner_selection_mthd"] == inner_selection]
            for classif in np.unique(df_inner["Classifiers"]):
                indices = df_inner[df_inner["Classifiers"] == classif]
                filtered_scores = indices[f"{self.config_rncv['outer_scoring']}"].values
                if num_features is not None:
                    filtered_features = indices["Selected_Features"].values
                mean_score = np.mean(filtered_scores)
                max_score = np.max(filtered_scores)
                std_score = np.std(filtered_scores)
                sem_score = sem(filtered_scores)
                median_score = np.median(filtered_scores)
                Numbers_of_Features = indices["Number_of_Features"].unique()[0]
                Way_of_Selection = indices["Way_of_Selection"].unique()[0]
                samples_classification_rates = np.zeros(len(self.y))
                for test_part in indices["Samples_counts"]:
                    samples_classification_rates=np.add(samples_classification_rates,test_part)
                samples_classification_rates /= rounds
                                
                results.append(
                    {
                        "Est": df_inner[df_inner["Classifiers"] == classif]["Estimator"].unique()[
                            0
                        ],
                        "Clf": classif,
                        "Hyp": df_inner[df_inner["Classifiers"] == classif][
                            "Hyperparameters"
                        ].values,
                        "Sel_way": Way_of_Selection,
                        "Fs_inner": 'none' if Way_of_Selection=='none' else feature_selection_method,
                        "Fs_num": Numbers_of_Features,
                        "Sel_feat": filtered_features
                        if num_features is not None
                        else None,
                        "Norm": normalization,
                        "Miss_vals": missing_values_method,
                        "In_cv": inner_splits,
                        "Out_cv": outer_splits,
                        "Rnds": rounds,
                        "Trials": n_trials,
                        "Class_blnc": self.config_rncv['class_balance'],
                        "In_scor": inner_scoring,
                        "Out_scor": outer_scoring,
                        "In_sel":inner_selection,
                        "Classif_rates": samples_classification_rates.tolist(),
                    }
                )

                results = _calc_metrics_stats(
                    self.config_rncv['extra_metrics'], results, indices
                )
                                            
        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)
        
        # Create a 'Results' directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Name
        final_dataset_name = _name_outputs(self.config_rncv, results_dir, self.csv_dir)  
            
        # Save the results to a CSV file of the outer scores for each classifier
        # if return_csv:
        statistics_dataframe = _return_csv(final_dataset_name, scores_dataframe, self.config_rncv['extra_metrics'], filter_csv, return_csv)
            
        # Manipulate the size of the plot to fit the number of features
        if num_features is not None:    
            # Plot histogram of features
            _histogram(scores_dataframe, final_dataset_name, freq_feat, self.config_rncv['clfs'], self.X.shape[1])
        
        # Plot box or violin plots of the outer cross-validation scores for all Inner_Selection methods
        if plot is not None:
            _plot_per_clf(scores_dataframe, plot, self.config_rncv['outer_scoring'], final_dataset_name)
            
        if info_to_db:
            # Add to database
            insert_to_db(scores_dataframe, self.config_rncv, database_name=self.database_name)
            
        return statistics_dataframe