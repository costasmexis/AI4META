import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from itertools import chain
from threadpoolctl import threadpool_limits
from src.utils.validation.dataclasses import ModelSelectionConfig
from src.data.process import DataProcessor
from src.database.manager import DatabaseManager
from src.utils.model_selection.ms_cv import _cv_loop
from src.utils.statistics.metrics_stats import _calc_metrics_stats
from src.utils.model_selection.output_config import _return_csv
from src.utils.plots.plots import _histogram, _plot_per_clf

class MLSelector(DataProcessor):
    """
    MLSelector class for machine learning model selection and evaluation.
    This class extends the DataProcessor class and provides methods for
    performing cross-validation, feature selection, and hyperparameter tuning.
    """
    def __init__(
        self,
        label: str,
        csv_dir: str,
        index_col: Optional[str] = None,
        normalization: Optional[str] = 'minmax',
        fs_method: Optional[str] = 'mrmr',
        inner_fs_method: Optional[str] = 'chi2',
        mv_method: Optional[str] = 'median',
        class_balance_method: Optional[str] = None,
        preprocess_mode: Optional[str] = 'ms',
        database_name: Optional[str] = None,
        inner_selection: Optional[List[str]] = ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"],
        extra_metrics: Optional[List[str]] = [
            'roc_auc', 'recall', 'matthews_corrcoef', 'accuracy', 'balanced_accuracy',
            'precision', 'f1', 'average_precision', 'specificity'
        ]
    ) -> None:
        """Initialize MLSelector instance."""
        # Initialize parent class
        super().__init__(
            label=label,
            csv_dir=csv_dir,
            index_col=index_col,
            normalization=normalization,
            fs_method=fs_method,
            mv_method=mv_method,
            inner_fs_method=inner_fs_method,
            class_balance_method=class_balance_method,
            preprocess_mode=preprocess_mode 
        )
        
        # Store database name
        self.database_name = database_name
        self.inner_selection = inner_selection
        self.extra_metrics = extra_metrics

        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for the class."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("MLSelector initialized with parameters:")
        self.logger.info(f"Label: {self.label}")
        self.logger.info(f"CSV Directory: {self.csv_dir}")
        self.logger.info(f"Index Column: {self.index_col}")
        self.logger.info(f"Normalization: {self.normalization}")
        self.logger.info(f"Feature Selection Method: {self.fs_method}")
        self.logger.info(f"Missing Values Method: {self.mv_method}")
        self.logger.info(f"Database Name: {self.database_name}")

    def modelselection(
        self, 
        model_selection_type: str = 'both',
        rounds: int = 10,
        exclude: Optional[List[str]] = None,
        search_on: Optional[List[str]] = None,
        num_features: Optional[Union[int, List[int]]] = None,
        return_csv: bool = True,
        plot: Optional[str] = 'box',            
        scoring: str = "roc_auc",
        inner_scoring: str = "matthews_corrcoef",
        splits: int = 5,
        inner_splits: int = 5,
        freq_feat: Optional[int] = None,
        sfm: bool = False,      
        info_to_db: bool = False,
        # filter_csv: Optional[Dict] = None, 
        parallel: str = "thread_per_round",
        n_trials: int = 100
    ):
        """
        Perform model selection using the specified parameters.
        
        Parameters
        ----------
        [parameter documentation omitted for brevity]
            
        Returns
        -------
        DataFrame
            Results of model selection
        """
        # Create and validate ModelSelectionConfig directly
        config = ModelSelectionConfig(
            model_selection_type=model_selection_type,
            normalization=self.normalization,
            feature_selection_type=self.fs_method,
            feature_selection_method=self.inner_fs_method,
            missing_values_method=self.mv_method,
            class_balance=self.class_balance_method,
            rounds=rounds,
            exclude=exclude,
            search_on=search_on,
            num_features=num_features,
            return_csv=return_csv,
            plot=plot,
            scoring=scoring,
            inner_scoring=inner_scoring,
            splits=splits,
            inner_splits=inner_splits,
            freq_feat=freq_feat,
            inner_selection=self.inner_selection,
            sfm=sfm,
            extra_metrics=self.extra_metrics,
            info_to_db=info_to_db,
            # filter_csv=filter_csv,
            parallel=parallel,
            n_trials=n_trials
        ).validate(self.X, self.csv_dir)   
        
        # Set up parallelization
        trial_indices = range(rounds)
        num_cores = multiprocessing.cpu_count()
        use_cores = min(num_cores, rounds)
        
        if config.parallel == "thread_per_round":
            avail_thr = 1
        else:
            avail_thr = max(1, num_cores // rounds)
        with threadpool_limits(limits=avail_thr):
            list_dfs = Parallel(n_jobs=use_cores)(
                delayed(_cv_loop)(self.X, self.y, config, i, avail_thr, self) for i in trial_indices
            )

        # Flatten results from parallel processing
        list_dfs_flat = list(chain.from_iterable(list_dfs))

        # Create results DataFrame
        df = pd.concat([pd.DataFrame(item) for item in list_dfs_flat], axis=0)

        results = []
        for  method in df['MS_Method'].unique():
            df_method = df[df["MS_Method"] == method]
            for inner in config.inner_selection:
                df_inner = df_method[df_method["Inner_selection_mthd"] == inner]
                for classif in np.unique(df_inner["Classifiers"]):
                    indices = df_inner[df_inner["Classifiers"] == classif]
                    filtered_scores = indices[f"{config.scoring}"]
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
                            "Est": str(df_inner[df_inner["Classifiers"] == classif]["Estimator"].unique()[0]),
                            "Clf": str(classif+"_"+inner+"_"+method),
                            "MS_Method": str(method),
                            "Hyp": np.array(df_inner[df_inner["Classifiers"] == classif]["Hyperparameters"].values),
                            "Sel_way": str(Way_of_Selection),
                            "Fs_inner": 'none' if Way_of_Selection == 'none' or Way_of_Selection == 'sfm' else str(config.feature_selection_method),
                            "Fs_num": int(Numbers_of_Features),
                            "Sel_feat": list(filtered_features),
                            "Norm": str(config.normalization),
                            "Miss_vals": str(config.missing_values_method),
                            "Inner_splits": int(config.inner_splits) if method == 'NestedCV' else 0,
                            "Splits": int(config.splits),
                            "Rnds": int(config.rounds),
                            "Trials": int(config.n_trials) if method == 'NestedCV' else 1,
                            "Cls_blnc": str(config.class_balance),
                            "Inner_scoring": str(config.inner_scoring) if method == 'NestedCV' else '',
                            "Scoring": str(config.scoring),
                            "Inner_selection": str(inner),
                            "Classif_rates": np.array(samples_classification_rates)
                        }
                    )
                    # Update results with calculated metrics
                    results = _calc_metrics_stats(config.extra_metrics, results, indices)

        print(f"Finished with {len(results)} models")
        scores_dataframe = pd.DataFrame(results)

        # Save to CSV if specified
        statistics_dataframe = _return_csv(config.dataset_csv_name, scores_dataframe, config.extra_metrics, return_csv)

        # Plot histogram if required
        if config.num_features is not None or config.num_features < self.X.shape[1]:    
            # Save the number that features selected
            denomination = len(scores_dataframe[scores_dataframe['Sel_way'] != 'none'])
            _histogram(scores_dataframe, config.dataset_histogram_name, freq_feat, denomination, self.X.shape[1])

        # Generate performance plots
        if plot is not None:
            _plot_per_clf(scores_dataframe, plot, config.scoring, config.dataset_plot_name)

        # Save results to database if specified
        if info_to_db:
            dbman = DatabaseManager(self.database_name)
            dbman.insert_experiment_data(scores_dataframe, config, database_name=self.database_name)

        return statistics_dataframe
