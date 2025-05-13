# In src/utils/validation/config_types.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
import re
import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import get_scorer_names
from src.constants.translators import DEFAULT_CONFIG_MS
from src.constants.translators import AVAILABLE_CLFS

@dataclass
class ModelSelectionConfig:
    """Data class for model selection configuration with built-in validation"""
    
    # Model selection parameters with default values
    model_selection_type: str = 'both'
    normalization: str = 'minmax'
    class_balance: str = None
    feature_selection_method: str = 'chi2'
    feature_selection_type: str = 'mrmr'
    missing_values_method: str = 'median'
    rounds: int = 10
    exclude: Optional[List[str]] = None
    search_on: Optional[List[str]] = None
    num_features: Optional[Union[int, List[int]]] = None
    return_csv: bool = True
    plot: Optional[str] = None
    scoring: str = "roc_auc"
    inner_scoring: str = "matthews_corrcoef"
    splits: int = 5
    inner_splits: int = 5
    freq_feat: Optional[int] = None
    inner_selection: List[str] = field(default_factory=lambda: ["validation_score", "one_sem", "gso_1", "gso_2", "one_sem_grd"])
    sfm: bool = False
    extra_metrics: List[str] = field(default_factory=lambda: ['roc_auc','recall', 'matthews_corrcoef', 'accuracy', 'balanced_accuracy',  
                           'precision', 'f1', 'average_precision', 'specificity'])
    info_to_db: bool = False
    filter_csv: Optional[Dict] = None
    parallel: str = "thread_per_round"
    n_trials: int = 100
    
    # Derived attributes (will be set during validation)
    clfs: Optional[List[str]] = None
    dataset_name: Optional[str] = None
    all_features: Optional[int] = None
    final_csv_name: Optional[str] = None
    dataset_csv_name: Optional[str] = None
    dataset_histogram_name: Optional[str] = None
    dataset_plot_name: Optional[str] = None
    
    # Validation helper variables
    _logger: Optional[logging.Logger] = field(default=None, repr=False)
    _logged_messages: set = field(default_factory=set, repr=False)
    
    def validate(self, X: pd.DataFrame, csv_dir: str) -> 'ModelSelectionConfig':
        """
        Validate this configuration instance against the dataset and return self.
        
        Parameters
        ----------
        X : DataFrame
            Input features for validation
        csv_dir : str
            Path to the CSV file
            
        Returns
        -------
        ModelSelectionConfig
            Self, after validation (for method chaining)
        """
        # Setup logging if not already done
        if self._logger is None:
            self._logger = logging.getLogger(__name__)
        
        # Validate model selection type
        self._validate_model_selection_type()
        
        # Validate number of features
        self._validate_num_features(X)
        
        # Validate scoring metrics
        self._validate_scoring_metrics()
        
        # Validate parallel option
        self._validate_parallel()
        
        # Configure classifiers
        self._configure_classifiers()
        
        # Extract dataset info
        self._extract_dataset_info(X, csv_dir)

        # Set result names
        self._set_result_names()
        
        return self
    
    def _log_once(self, message: str, level: str = 'warning') -> None:
        """Log a message only if it hasn't been logged before."""
        if message not in self._logged_messages:
            if level == 'warning':
                self._logger.warning(message)
            elif level == 'error':
                self._logger.error(message)
            elif level == 'info':
                self._logger.info(message)
            self._logged_messages.add(message)
    
    def _validate_model_selection_type(self) -> None:
        """Validate model selection type"""
        valid_model_types = ["rcv_accel", "rnested_cv", "both"]
        if self.model_selection_type not in valid_model_types:
            self._log_once(f"Invalid model selection type: {self.model_selection_type}. Using default: both")
            self.model_selection_type = 'both'
    
    def _validate_num_features(self, X: pd.DataFrame) -> None:
        """Validate number of features"""
        if isinstance(self.num_features, int):
            self.num_features = [self.num_features]
        elif self.num_features is None:
            self.num_features = [X.shape[1]]
        elif not isinstance(self.num_features, list):
            self._log_once("num_features must be an integer, list, or None. Using all features.")
            self.num_features = [X.shape[1]]
    
    def _validate_scoring_metrics(self) -> None:
        """Validate scoring metrics"""
        valid_scorers = list(get_scorer_names()) + ["specificity"]
        
        if self.scoring not in valid_scorers:
            raise ValueError(f"Invalid scoring metric: {self.scoring}. "
                           f"Valid options: {valid_scorers}")
        
        if self.inner_scoring not in valid_scorers:
            raise ValueError(f"Invalid inner scoring metric: {self.inner_scoring}. "
                            f"Valid options: {valid_scorers}")
            
        # Ensure outer_scoring is in extra_metrics
        if self.scoring not in self.extra_metrics:
            self.extra_metrics.insert(0, self.scoring)
    
    def _validate_parallel(self) -> None:
        """Validate parallel option"""
        valid_parallel_options = ["thread_per_round", "freely_parallel"]
        if self.parallel not in valid_parallel_options:
            self._log_once(f"Invalid parallel value: {self.parallel}. Using default: thread_per_round")
            self.parallel = "thread_per_round"

    def _configure_classifiers(self) -> None:
        """Configure classifier selection based on include/exclude lists"""
        if self.exclude is None:
            self.exclude = []
            
        if self.search_on:
            self.clfs = [clf for clf in self.search_on if clf in AVAILABLE_CLFS]
        else:
            self.clfs = [clf for clf in AVAILABLE_CLFS if clf not in self.exclude]
    
    def _extract_dataset_info(self, X: pd.DataFrame, csv_dir: str) -> None:
        """Extract and store dataset-related information"""
        # Extract dataset name from path
        pattern = r"[^/]+(?=\.csv)"
        match = re.search(pattern, csv_dir)
        self.dataset_name = match.group() if match else ''
        
        # Set features information
        self.all_features = X.shape[1]

    def _extract_final_name(self) -> str:
        """
        Extract the final name for plots, CSVs, etc., including non-default parameter values.
        
        Returns
        -------
        str
            Formatted name including dataset name and non-default parameters
        """
        # Start with dataset name
        final_name = self.dataset_name
        
        # Get default values from constants
        defaults = DEFAULT_CONFIG_MS

        # Add non-default configurations
        for param, value in defaults.items():
            if param in vars(self) and getattr(self, param) != value:
                final_name += f"_{param}_{getattr(self, param)}"

        # Add timestamp to ensure uniqueness
        final_name += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Add the type of model selection
        if self.model_selection_type == "both":
            final_name += "_MS_BOTH"
        elif self.model_selection_type == "rcv_accel":
            final_name += "_MS_RCV"
        elif self.model_selection_type == "rnested_cv":
            final_name += "_MS_RCV_NESTED"

        return final_name

    def _set_result_names(self) -> None:
        # Get the base name
        base_name = self._extract_final_name()
        
        # CSV results directory
        results_csv_dir = "results/csv/"
        os.makedirs(results_csv_dir, exist_ok=True)
        self.dataset_csv_name = os.path.join(results_csv_dir, f"{base_name}.csv")
        
        # Image results directory
        results_image_dir = "results/images/"
        os.makedirs(results_image_dir, exist_ok=True)
        self.dataset_plot_name = os.path.join(results_image_dir, f"{base_name}_plot.png")
        
        # Histogram results directory
        results_hist_dir = "results/histograms/"
        os.makedirs(results_hist_dir, exist_ok=True)
        self.dataset_histogram_name = os.path.join(results_hist_dir, f"{base_name}_histogram.png")
        